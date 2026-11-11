// Prevent Windows min/max macros from conflicting with C++ functions.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "kmeans_sequential.h"
#include "kmeans_openmp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>

using namespace std;

// Select the implementations and datasets before compiling.
constexpr bool RUN_SEQUENTIAL = true;
constexpr bool RUN_OPENMP = true;
constexpr int OPENMP_THREADS = 4;
constexpr bool RUN_ALL_DATASETS = false;
constexpr const char* DATASET_DIRECTORY = "datasets";
constexpr const char* SELECTED_DATASET = "blobs_N100000_D16.txt";

constexpr int K = 10;
constexpr int MAX_ITERATIONS = 100;
constexpr int WARMUP_RUNS = 2;
constexpr int MEASURED_RUNS = 30;
constexpr bool PIN_TO_CORE = true;
constexpr int LOGICAL_CPU = 2;

static_assert(RUN_SEQUENTIAL || RUN_OPENMP, "Enable at least one implementation.");
static_assert(WARMUP_RUNS >= 0, "WARMUP_RUNS cannot be negative.");
static_assert(OPENMP_THREADS > 0, "OPENMP_THREADS must be positive.");
static_assert(MEASURED_RUNS > 0, "MEASURED_RUNS must be positive.");

static_assert(LOGICAL_CPU >= 0 &&
              LOGICAL_CPU < static_cast<int>(sizeof(DWORD_PTR) * 8),
              "Invalid logical CPU index.");


vector<vector<double>> load_dataset(const filesystem::path& filename) {
    ifstream file(filename);

    if (!file) {
        throw runtime_error("Cannot open the dataset file. File: " + filename.string());
    }

    int n;
    int dimensions;

    if (!(file >> n >> dimensions) || n <= 0 || dimensions <= 0) {
        throw runtime_error("Invalid dataset header. File: " + filename.string());
    }

    vector<vector<double>> points(
        n, vector<double>(dimensions)
    );

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dimensions; d++) {
            if (!(file >> points[i][d])) {
                throw runtime_error("Invalid or incomplete dataset. File: " + filename.string());
            }
        }
    }

    return points;
}

vector<filesystem::path> select_datasets() {
    const filesystem::path directory = DATASET_DIRECTORY;

    if (!RUN_ALL_DATASETS) {
        return {directory / SELECTED_DATASET};
    }

    vector<filesystem::path> paths;
    for (const auto& entry : filesystem::directory_iterator(directory)) {
        const auto& path = entry.path();
        if (entry.is_regular_file() &&
            path.extension() == ".txt" &&
            path.filename().string().rfind("blobs_N", 0) == 0) {
            paths.push_back(path);
        }
    }

    if (paths.empty()) {
        throw runtime_error("No blob datasets found in " + directory.string());
    }

    sort(paths.begin(), paths.end());
    return paths;
}

using Centroids = vector<vector<double>>;
using CentroidRuns = vector<Centroids>;

template <typename KMeansFunction>
CentroidRuns benchmark(
    const char* implementation_name,
    KMeansFunction run_kmeans,
    const vector<vector<double>>& points
) {
    const int n = static_cast<int>(points.size());
    const int dimensions = static_cast<int>(points.front().size());

    // Use the first K points as fixed initial centroids.
    vector<vector<double>> initial_centroids(
        K, vector<double>(dimensions)
    );

    for (int c = 0; c < K; c++) {
        initial_centroids[c] = points[c];
    }

    // Allocate working memory outside the measured region.
    vector<vector<double>> centroids(
        K, vector<double>(dimensions)
    );
    vector<int> labels(n, -1);

    vector<double> times(MEASURED_RUNS);
    vector<int> iteration_counts(MEASURED_RUNS);
    CentroidRuns measured_centroids(MEASURED_RUNS, centroids);

    cout << "\nImplementation: " << implementation_name << "\n";
    cout << "Running " << WARMUP_RUNS << " warm-ups and "
         << MEASURED_RUNS << " measured runs..."
         << endl;

    for (int run = 0; run < WARMUP_RUNS + MEASURED_RUNS; run++) {
        // Restore the same initial state before every run.
        for (int c = 0; c < K; c++) {
            for (int d = 0; d < dimensions; d++) {
                centroids[c][d] = initial_centroids[c][d];
            }
        }

        for (int i = 0; i < n; i++) {
            labels[i] = -1;
        }

        // Measure only the K-Means computation.
        auto start = chrono::steady_clock::now();

        int iterations = run_kmeans(
            centroids,
            labels,
            MAX_ITERATIONS
        );

        auto end = chrono::steady_clock::now();

        if (run >= WARMUP_RUNS) {
            int index = run - WARMUP_RUNS;

            times[index] =
                chrono::duration<double>(end - start).count();

            iteration_counts[index] = iterations;

            // Save the result after stopping the timer.
            measured_centroids[index] = centroids;
        }
    }

    // Sort a copy of the measured execution times.
    vector<double> sorted_times = times;
    sort(sorted_times.begin(), sorted_times.end());

    // Calculate the median for either an even or odd number of runs.
    int middle = static_cast<int>(sorted_times.size()) / 2;
    double median;

    if (sorted_times.size() % 2 == 0) {
        median = (sorted_times[middle - 1] + sorted_times[middle]) / 2.0;
    } else {
        median = sorted_times[middle];
    }

    // Print results outside the measured region.
    cout << fixed << setprecision(6);

    for (int run = 0; run < MEASURED_RUNS; run++) {
        cout << "Run " << run + 1
             << ": " << times[run] << " s"
             << ", iterations: " << iteration_counts[run]
             << "\n";
    }

    cout << "\nMedian time: " << median << " s\n";
    cout << "Minimum time: " << sorted_times.front() << " s\n";
    cout << "Maximum time: " << sorted_times.back() << " s\n";

    return measured_centroids;
}

void print_centroid_differences(
    const char* implementation_name,
    const CentroidRuns& runs,
    const Centroids& reference
) {
    double worst_rmse = 0.0;
    double worst_absolute_difference = 0.0;
    std::size_t worst_rmse_run = 1;
    std::size_t worst_absolute_run = 1;

    for (std::size_t run = 0; run < runs.size(); run++) {
        double squared_error_sum = 0.0;
        std::size_t coordinate_count = 0;

        for (std::size_t c = 0; c < reference.size(); c++) {
            for (std::size_t d = 0; d < reference[c].size(); d++) {
                const double difference = runs[run][c][d] - reference[c][d];

                if (!std::isfinite(difference)) {
                    throw runtime_error("Non-finite centroid difference.");
                }

                squared_error_sum += difference * difference;
                coordinate_count++;

                const double absolute_difference = std::abs(difference);
                if (absolute_difference > worst_absolute_difference) {
                    worst_absolute_difference = absolute_difference;
                    worst_absolute_run = run + 1;
                }
            }
        }

        // Compare RMSE per coordinate across measured runs.
        const double rmse = std::sqrt(
            squared_error_sum / static_cast<double>(coordinate_count)
        );
        if (rmse > worst_rmse) {
            worst_rmse = rmse;
            worst_rmse_run = run + 1;
        }
    }

    cout << implementation_name << " (" << runs.size() << " measured runs)\n";
    cout << scientific << setprecision(6);
    cout << "Worst RMSE per coordinate: " << worst_rmse
         << " (run " << worst_rmse_run << ")\n";
    cout << "Maximum absolute difference: " << worst_absolute_difference
         << " (run " << worst_absolute_run << ")\n";
    cout << defaultfloat;
}

int main() {
    // Single-CPU affinity applies only to sequential-only benchmarks.
    if (PIN_TO_CORE && !RUN_OPENMP) {
        DWORD_PTR affinity_mask = DWORD_PTR{1} << LOGICAL_CPU;

        if (!SetProcessAffinityMask(GetCurrentProcess(), affinity_mask)) {
            cerr << "Cannot set CPU affinity. Windows error: "
                 << GetLastError() << "\n";
            return 1;
        }

        cout << "Process restricted to logical CPU "
             << LOGICAL_CPU << ".\n";
    } else {
        cout << "CPU affinity not explicitly restricted by this program.\n";
    }
    try {
        if (RUN_OPENMP) {
            omp_set_dynamic(0); // Keep OpenMP from automatically changing the number of threads.
            omp_set_num_threads(OPENMP_THREADS);
            cout << "Requested OpenMP threads: " << OPENMP_THREADS << "\n";
        }
        for (const auto& dataset_path : select_datasets()) {
            // Load each dataset once before benchmarking either implementation.
            const auto points = load_dataset(dataset_path);
            const int n = static_cast<int>(points.size());

            if (K <= 0 || K > n || MAX_ITERATIONS <= 0) {
                throw runtime_error("Invalid K or iteration limit.");
            }

            const int dimensions = static_cast<int>(points.front().size());

            // Store all points in one contiguous buffer, keeping point order.
            vector<double> flat_points;

            if (RUN_OPENMP) {
                flat_points.resize(static_cast<std::size_t>(n) * dimensions);

                for (int i = 0; i < n; i++) {
                    for (int d = 0; d < dimensions; d++) {
                        flat_points[static_cast<std::size_t>(i) * dimensions + d] =
                            points[i][d];
                    }
                }
            }

            cout << "\nDataset: " << dataset_path.generic_string() << "\n";
            cout << "Points: " << n << "\n";
            cout << "Dimensions: " << points.front().size() << "\n";
            cout << "Clusters: " << K << "\n";
            cout << "Maximum iterations: " << MAX_ITERATIONS << "\n";

            CentroidRuns sequential_runs;
            CentroidRuns openmp_runs;

            if (RUN_SEQUENTIAL) {
                // Prepare the sequential working arrays before timing.
                Centroids sums(K, vector<double>(dimensions, 0.0));
                vector<int> counts(K, 0);

                auto run_sequential = [&points, &sums, &counts](
                    Centroids& centroids,
                    vector<int>& labels,
                    int max_iterations
                ) {
                    return kmeans(
                        points, centroids, labels, sums, counts, max_iterations
                    );
                };

                sequential_runs = benchmark("Sequential", run_sequential, points);
            }

            if (RUN_OPENMP) {
                auto run_openmp = [&flat_points, n, dimensions](
                    Centroids& centroids,
                    vector<int>& labels,
                    int max_iterations
                ) {
                    return kmeans_openmp(
                        flat_points, n, dimensions,
                        centroids, labels, max_iterations
                    );
                };

                openmp_runs = benchmark("OpenMP", run_openmp, points);
            }

            // Use one reference for every measured run of this dataset.
            const auto& reference = RUN_SEQUENTIAL
                ? sequential_runs.front()
                : openmp_runs.front();
            cout << "\nCentroid comparison (reference: first "
                 << (RUN_SEQUENTIAL ? "sequential" : "OpenMP")
                 << " measured run)\n";

            if (RUN_SEQUENTIAL) {
                print_centroid_differences("Sequential", sequential_runs, reference);
            }
            if (RUN_OPENMP) {
                print_centroid_differences("OpenMP", openmp_runs, reference);
            }
        }

        return 0;
    } catch (const exception& error) {
        cerr << error.what() << "\n";
        return 1;
    }
}

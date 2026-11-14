// Prevent Windows min/max macros from conflicting with C++ functions.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "kmeans_sequential.h"
#include "kmeans_openmp.h"
#include "kmeans_openmp_soa.h"
#include "kmeans_openmp_soa_dynamic.h"
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
#include <locale>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

using namespace std;

// Select the implementations and datasets before compiling.
constexpr bool RUN_SEQUENTIAL = true;
constexpr bool RUN_OPENMP = false; // this refers to AoS implementation
constexpr bool RUN_OPENMP_SOA = true; // Static scheduling.
constexpr bool RUN_OPENMP_SOA_DYNAMIC = true;
constexpr bool RUN_ANY_SOA = RUN_OPENMP_SOA || RUN_OPENMP_SOA_DYNAMIC;
constexpr bool RUN_ANY_OPENMP = RUN_OPENMP || RUN_ANY_SOA;
constexpr int OPENMP_THREADS = 24;
constexpr bool RUN_ALL_THREAD_CONFIGURATIONS = true;
constexpr int THREAD_COUNTS[] = {1, 2, 4, 6, 8, 16, 24, 32};
constexpr bool MEASURE_ONE_THREAD_BASELINES = true;
constexpr bool RUN_ALL_DATASETS = true;
constexpr const char* DATASET_DIRECTORY = "datasets";
constexpr const char* SELECTED_DATASET = "blobs_N100000_D65.txt";

constexpr int K = 30;
constexpr bool RUN_ALL_CLUSTER_COUNTS = true;
constexpr int CLUSTER_COUNTS[] = {10, 20, 30};
constexpr int MAX_ITERATIONS = 300;
constexpr int WARMUP_RUNS = 2;
constexpr int MEASURED_RUNS = 10;
constexpr bool RESTRICT_CPU_AFFINITY = true;
constexpr const char* RESULTS_DIRECTORY = "results";

static_assert(RUN_SEQUENTIAL || RUN_ANY_OPENMP, "Enable at least one implementation.");
static_assert(WARMUP_RUNS >= 0, "WARMUP_RUNS cannot be negative.");
static_assert(OPENMP_THREADS > 0, "OPENMP_THREADS must be positive.");
static_assert(MEASURED_RUNS > 0, "MEASURED_RUNS must be positive.");

struct CpuConfiguration {
    DWORD_PTR mask;
    const char* description;
};

// CPU numbering verified on this machine; not a universal Intel mapping.
constexpr CpuConfiguration select_cpu_configuration(int threads) {
    switch (threads) {
        case 1:  return {0x00000001ULL, "1 physical P-core"};
        case 2:  return {0x00000005ULL, "2 physical P-cores"};
        case 4:  return {0x00000055ULL, "4 physical P-cores"};
        case 6:  return {0x00000555ULL, "6 physical P-cores"};
        case 8:  return {0x00005555ULL, "8 physical P-cores"};
        case 16: return {0x0000FFFFULL, "8 P-cores with Hyper-Threading"};
        case 24: return {0xFFFF5555ULL, "24 physical cores: 8 P + 16 E"};
        case 32: return {0xFFFFFFFFULL, "All 32 logical processors"};
        default: return {0, "Unsupported configuration"};
    }
}

constexpr CpuConfiguration CPU_CONFIGURATION =
    select_cpu_configuration(OPENMP_THREADS);

static_assert(sizeof(DWORD_PTR) == 8, "Use a 64-bit toolchain.");

static_assert(
    !RUN_ANY_OPENMP || RUN_ALL_THREAD_CONFIGURATIONS || CPU_CONFIGURATION.mask != 0,
    "Supported thread counts: 1, 2, 4, 6, 8, 16, 24, 32."
);

vector<int> select_thread_counts() {
    vector<int> selected;
    if (RUN_ALL_THREAD_CONFIGURATIONS) {
        for (int threads : THREAD_COUNTS) {
            selected.push_back(threads);
        }
    } else {
        selected.push_back(OPENMP_THREADS);
    }

    vector<int> checked;
    for (int threads : selected) {
        if (select_cpu_configuration(threads).mask == 0) {
            throw runtime_error("Unsupported value in the thread configurations.");
        }
        if (find(checked.begin(), checked.end(), threads) != checked.end()) {
            throw runtime_error("Duplicate value in the thread configurations.");
        }
        checked.push_back(threads);
    }
    if (selected.empty()) {
        throw runtime_error("Select at least one thread configuration.");
    }
    return selected;
}

vector<int> select_cluster_counts() {
    vector<int> selected;
    if (RUN_ALL_CLUSTER_COUNTS) {
        for (int clusters : CLUSTER_COUNTS) {
            selected.push_back(clusters);
        }
    } else {
        selected.push_back(K);
    }

    vector<int> checked;
    for (int clusters : selected) {
        if (clusters <= 0) {
            throw runtime_error("The number of clusters must be positive.");
        }
        if (find(checked.begin(), checked.end(), clusters) != checked.end()) {
            throw runtime_error("Duplicate value in the cluster configurations.");
        }
        checked.push_back(clusters);
    }
    if (selected.empty()) {
        throw runtime_error("Select at least one cluster configuration.");
    }
    return selected;
}


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

struct TimeStatistics {
    double median;
    double mean;
    double standard_deviation;
    double minimum;
    double maximum;
};

struct BenchmarkResult {
    TimeStatistics statistics{};
    vector<double> times;
    vector<int> iteration_counts;
    CentroidRuns centroids;
};

struct CentroidDifference {
    double worst_rmse;
    double maximum_absolute_difference;
};

struct PerformanceMetrics {
    optional<double> speedup_vs_sequential;
    optional<double> relative_speedup;
    optional<double> relative_efficiency;
    optional<bool> iteration_range_matches_sequential;
    optional<bool> iteration_range_matches_one_thread;
};

TimeStatistics calculate_statistics(const vector<double>& times) {
    if (times.empty()) {
        throw runtime_error("Cannot calculate statistics without measurements.");
    }

    vector<double> sorted_times = times;
    sort(sorted_times.begin(), sorted_times.end());
    const size_t middle = sorted_times.size() / 2;
    const double median = sorted_times.size() % 2 == 0
        ? (sorted_times[middle - 1] + sorted_times[middle]) / 2.0
        : sorted_times[middle];

    const double mean = accumulate(times.begin(), times.end(), 0.0)
        / static_cast<double>(times.size());

    double squared_deviations = 0.0;
    for (double time : times) {
        const double deviation = time - mean;
        squared_deviations += deviation * deviation;
    }

    // Use the sample standard deviation; one measurement has no estimated spread.
    const double standard_deviation = times.size() > 1
        ? sqrt(squared_deviations / static_cast<double>(times.size() - 1))
        : 0.0;

    return {median, mean, standard_deviation, sorted_times.front(), sorted_times.back()};
}

pair<int, int> iteration_range(const BenchmarkResult& result) {
    const auto limits = minmax_element(
        result.iteration_counts.begin(), result.iteration_counts.end()
    );
    return {*limits.first, *limits.second};
}

PerformanceMetrics calculate_performance(
    const BenchmarkResult& result,
    const BenchmarkResult* sequential_reference,
    const BenchmarkResult* one_thread_reference,
    int threads
) {
    PerformanceMetrics metrics;
    const double median = result.statistics.median;

    if (sequential_reference != nullptr) {
        metrics.iteration_range_matches_sequential =
            iteration_range(result) == iteration_range(*sequential_reference);
        if (median > 0.0 && sequential_reference->statistics.median > 0.0) {
            metrics.speedup_vs_sequential = sequential_reference->statistics.median / median;
        }
    }

    if (one_thread_reference != nullptr) {
        metrics.iteration_range_matches_one_thread =
            iteration_range(result) == iteration_range(*one_thread_reference);
        if (median > 0.0 && one_thread_reference->statistics.median > 0.0) {
            metrics.relative_speedup = one_thread_reference->statistics.median / median;
            metrics.relative_efficiency = *metrics.relative_speedup / threads;
        }
    }

    return metrics;
}

template <typename KMeansFunction>
BenchmarkResult benchmark(
    const char* implementation_name,
    KMeansFunction run_kmeans,
    const vector<vector<double>>& points,
    int k
) {
    const int n = static_cast<int>(points.size());
    const int dimensions = static_cast<int>(points.front().size());

    // Use the first k points as fixed initial centroids.
    vector<vector<double>> initial_centroids(
        k, vector<double>(dimensions)
    );

    for (int c = 0; c < k; c++) {
        initial_centroids[c] = points[c];
    }

    // Allocate working memory outside the measured region.
    vector<vector<double>> centroids(
        k, vector<double>(dimensions)
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
        for (int c = 0; c < k; c++) {
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

    // Calculate all statistics after the measured runs have finished.
    const TimeStatistics statistics = calculate_statistics(times);

    // Print results outside the measured region.
    cout << fixed << setprecision(6);

    for (int run = 0; run < MEASURED_RUNS; run++) {
        cout << "Run " << run + 1
             << ": " << times[run] << " s"
             << ", iterations: " << iteration_counts[run]
             << "\n";
    }

    cout << "\nMedian time: " << statistics.median << " s\n";
    cout << "Minimum time: " << statistics.minimum << " s\n";
    cout << "Maximum time: " << statistics.maximum << " s\n";

    cout << "Mean time: " << statistics.mean << " s\n";
    cout << "Sample standard deviation: " << statistics.standard_deviation << " s\n";
    if (*max_element(iteration_counts.begin(), iteration_counts.end()) >= MAX_ITERATIONS) {
        cout << "NOTE: a run reached the iteration limit; convergence at that limit is not distinguished.\n";
    }

    return {statistics, std::move(times), std::move(iteration_counts), std::move(measured_centroids)};
}

CentroidDifference print_centroid_differences(
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
    return {worst_rmse, worst_absolute_difference};
}

// All functions below run outside the measured K-Means call.
void configure_execution(int threads, DWORD_PTR original_mask) {
    const CpuConfiguration configuration = select_cpu_configuration(threads);
    if (configuration.mask == 0) {
        throw runtime_error("Unsupported thread configuration.");
    }

    const DWORD_PTR mask = RESTRICT_CPU_AFFINITY ? configuration.mask : original_mask;
    if ((mask & original_mask) != mask) {
        throw runtime_error("The selected CPUs are outside the process's initial allowed CPUs.");
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask)) {
        throw runtime_error("Cannot set CPU affinity. Windows error: " + to_string(GetLastError()));
    }

    omp_set_dynamic(0); // Keep OpenMP from automatically changing the number of threads.
    omp_set_num_threads(threads);

    cout << "\nExecution configuration: "
         << (RESTRICT_CPU_AFFINITY ? configuration.description : "Windows scheduling without an added affinity restriction")
         << "\nRequested threads: " << threads << "\nAllowed logical CPUs:";
    for (int cpu = 0; cpu < static_cast<int>(sizeof(DWORD_PTR) * 8); cpu++) {
        if ((mask & (DWORD_PTR{1} << cpu)) != 0) {
            cout << " " << cpu;
        }
    }
    cout << "\n";
}

filesystem::path create_result_directory() {
    SYSTEMTIME time{};
    GetLocalTime(&time);
    ostringstream name;
    name << "benchmark_" << setfill('0')
         << setw(4) << time.wYear << setw(2) << time.wMonth << setw(2) << time.wDay
         << "_" << setw(2) << time.wHour << setw(2) << time.wMinute << setw(2) << time.wSecond
         << "_" << setw(3) << time.wMilliseconds << "_" << GetCurrentProcessId();

    const filesystem::path root = RESULTS_DIRECTORY;
    filesystem::create_directories(root);
    const filesystem::path base = root / name.str();
    filesystem::path directory = base;
    int suffix = 0;
    while (!filesystem::create_directory(directory)) {
        directory = base.string() + "_" + to_string(++suffix);
    }
    return directory;
}

string csv_text(const string& text) {
    string escaped = "\"";
    for (char character : text) {
        if (character == '"') {
            escaped += '"';
        }
        escaped += character;
    }
    return escaped + '"';
}

void write_csv_parameters(
    ostream& output,
    const filesystem::path& dataset,
    int n,
    int dimensions,
    int k,
    const char* implementation,
    const char* role,
    int threads,
    DWORD_PTR original_mask
) {
    const CpuConfiguration configuration = select_cpu_configuration(threads);
    const DWORD_PTR mask = RESTRICT_CPU_AFFINITY ? configuration.mask : original_mask;
    output << csv_text(dataset.generic_string()) << ',' << n << ',' << dimensions << ',' << k << ','
           << csv_text(implementation) << ',' << csv_text(role) << ',' << threads << ','
           << RESTRICT_CPU_AFFINITY << ',' << mask << ','
           << csv_text(RESTRICT_CPU_AFFINITY ? configuration.description : "Unrestricted") << ','
           << WARMUP_RUNS << ',' << MEASURED_RUNS << ',' << MAX_ITERATIONS;
}

template <typename Value>
void write_optional(ostream& output, const optional<Value>& value) {
    if (value.has_value()) {
        output << *value;
    }
}

void print_performance(const PerformanceMetrics& metrics) {
    cout << fixed << setprecision(6);
    if (metrics.speedup_vs_sequential.has_value()) {
        cout << "Speedup vs sequential: " << *metrics.speedup_vs_sequential << "\n";
    }
    if (metrics.relative_speedup.has_value()) {
        cout << "Relative speedup vs same version at 1 thread: " << *metrics.relative_speedup << "\n";
        cout << "Relative parallel efficiency: " << *metrics.relative_efficiency
             << " (" << *metrics.relative_efficiency * 100.0 << "%)\n";
    }
    if ((metrics.iteration_range_matches_sequential.has_value() && !*metrics.iteration_range_matches_sequential)
        || (metrics.iteration_range_matches_one_thread.has_value() && !*metrics.iteration_range_matches_one_thread)) {
        cout << "NOTE: iteration counts differ from a reference; consider this when interpreting speedup.\n";
    }
}

void save_result(
    ofstream& runs_csv,
    ofstream& summary_csv,
    const filesystem::path& dataset,
    int n,
    int dimensions,
    int k,
    const char* implementation,
    const char* role,
    int threads,
    DWORD_PTR original_mask,
    const BenchmarkResult& result,
    const PerformanceMetrics& metrics,
    const CentroidDifference& difference
) {
    for (size_t run = 0; run < result.times.size(); run++) {
        write_csv_parameters(runs_csv, dataset, n, dimensions, k, implementation, role, threads, original_mask);
        runs_csv << ',' << run + 1 << ',' << result.times[run] << ',' << result.iteration_counts[run]
                 << ',' << (result.iteration_counts[run] >= MAX_ITERATIONS) << '\n';
    }

    const auto& statistics = result.statistics;
    const auto limits = iteration_range(result);
    write_csv_parameters(summary_csv, dataset, n, dimensions, k, implementation, role, threads, original_mask);
    summary_csv << ',' << statistics.median << ',' << statistics.mean << ',' << statistics.standard_deviation
                << ',' << statistics.minimum << ',' << statistics.maximum << ',' << limits.first << ',' << limits.second
                << ',' << (limits.second >= MAX_ITERATIONS) << ',';
    write_optional(summary_csv, metrics.speedup_vs_sequential);
    summary_csv << ',';
    write_optional(summary_csv, metrics.relative_speedup);
    summary_csv << ',';
    write_optional(summary_csv, metrics.relative_efficiency);
    summary_csv << ',';
    write_optional(summary_csv, metrics.iteration_range_matches_sequential);
    summary_csv << ',';
    write_optional(summary_csv, metrics.iteration_range_matches_one_thread);
    summary_csv << ',' << difference.worst_rmse << ',' << difference.maximum_absolute_difference << '\n';
    runs_csv.flush();
    summary_csv.flush();
}

int main() {
    try {
        DWORD_PTR original_mask = 0;
        DWORD_PTR system_mask = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &original_mask, &system_mask)) {
            throw runtime_error("Cannot read the initial CPU affinity.");
        }
        const vector<int> thread_counts = select_thread_counts();
        const vector<int> cluster_counts = select_cluster_counts();
        const bool one_thread_selected =
            find(thread_counts.begin(), thread_counts.end(), 1) != thread_counts.end();

        const filesystem::path result_directory = create_result_directory();
        ofstream runs_csv;
        ofstream summary_csv;
        runs_csv.exceptions(ios::failbit | ios::badbit);
        summary_csv.exceptions(ios::failbit | ios::badbit);
        runs_csv.open(result_directory / "runs.csv");
        summary_csv.open(result_directory / "summary.csv");
        runs_csv.imbue(locale::classic());
        summary_csv.imbue(locale::classic());
        runs_csv << setprecision(17);
        summary_csv << setprecision(17);

        const char* common_columns = "dataset,N,D,K,implementation,role,requested_threads,affinity_restricted,allowed_cpu_mask,cpu_configuration,warmups,measured_runs,max_iterations";
        runs_csv << common_columns << ",run,wall_seconds,iterations,iteration_limit_reached\n";
        summary_csv << common_columns
            << ",median_s,mean_s,sample_stddev_s,min_s,max_s,min_iterations,max_run_iterations,any_run_at_iteration_limit"
            << ",speedup_vs_sequential,relative_speedup,relative_efficiency"
            << ",iteration_range_matches_sequential,iteration_range_matches_one_thread,worst_rmse,max_centroid_difference\n";

        cout << "Results directory: " << filesystem::absolute(result_directory).string() << "\n";
        cout << "Cluster configurations:";
        for (int clusters : cluster_counts) {
            cout << " " << clusters;
        }
        cout << "\n";
        if (RUN_ANY_OPENMP) {
            cout << "OpenMP thread configurations:";
            for (int threads : thread_counts) {
                cout << " " << threads;
            }
            cout << "\n";
        }
        if (!RUN_SEQUENTIAL && RUN_ANY_OPENMP) {
            cout << "Sequential disabled: speedup vs sequential will be unavailable.\n";
        }
        if (!MEASURE_ONE_THREAD_BASELINES && RUN_ANY_OPENMP && !one_thread_selected) {
            cout << "One-thread references disabled: relative speedup and efficiency will be unavailable.\n";
        }

        for (const auto& dataset_path : select_datasets()) {
            const auto points = load_dataset(dataset_path);
            const int n = static_cast<int>(points.size());
            const int dimensions = static_cast<int>(points.front().size());
            if (MAX_ITERATIONS <= 0) {
                throw runtime_error("Invalid iteration limit.");
            }
            for (int clusters : cluster_counts) {
                if (clusters > n) {
                    throw runtime_error("K exceeds the number of points in " + dataset_path.string());
                }
            }

            // Prepare both layouts outside all measured regions.
            vector<double> flat_points;
            if (RUN_OPENMP) {
                flat_points.resize(static_cast<size_t>(n) * dimensions);
                for (int i = 0; i < n; i++) {
                    for (int d = 0; d < dimensions; d++) {
                        flat_points[static_cast<size_t>(i) * dimensions + d] = points[i][d];
                    }
                }
            }

            vector<double> soa_points;
            if (RUN_ANY_SOA) {
                soa_points.resize(static_cast<size_t>(n) * dimensions);
                for (int d = 0; d < dimensions; d++) {
                    for (int i = 0; i < n; i++) {
                        soa_points[static_cast<size_t>(d) * n + i] = points[i][d];
                    }
                }
            }

            cout << "\nDataset: " << dataset_path.generic_string()
                 << "\nPoints: " << n << "\nDimensions: " << dimensions
                 << "\nMaximum iterations: " << MAX_ITERATIONS << "\n";

            for (int k : cluster_counts) {
                cout << "\nClusters: " << k << "\n";

                BenchmarkResult sequential_result;
                Centroids reference_centroids;
                string reference_name;

                // Summaries, numerical comparisons and CSV output are outside the timer.
                auto record_result = [&](const char* implementation, const char* role, int threads,
                                         const BenchmarkResult& result, const BenchmarkResult* one_thread_reference) {
                    if (reference_centroids.empty()) {
                        reference_centroids = result.centroids.front();
                        reference_name = implementation;
                    }

                    cout << "\nCentroid reference: first measured run of " << reference_name << "\n";
                    const CentroidDifference difference = print_centroid_differences(
                        implementation, result.centroids, reference_centroids
                    );
                    const PerformanceMetrics metrics = calculate_performance(
                        result, RUN_SEQUENTIAL ? &sequential_result : nullptr,
                        one_thread_reference, threads
                    );
                    print_performance(metrics);
                    save_result(runs_csv, summary_csv, dataset_path, n, dimensions, k, implementation, role,
                                threads, original_mask, result, metrics, difference);
                };

                if (RUN_SEQUENTIAL) {
                    // The baseline is pinned to CPU 0 when affinity restriction is enabled.
                    configure_execution(1, original_mask);
                    Centroids sums(k, vector<double>(dimensions, 0.0));
                    vector<int> counts(k, 0);
                    auto run_sequential = [&points, &sums, &counts](
                        Centroids& centroids, vector<int>& labels, int max_iterations
                    ) {
                        return kmeans(points, centroids, labels, sums, counts, max_iterations);
                    };
                    sequential_result = benchmark("Sequential", run_sequential, points, k);
                    record_result("Sequential", "sequential_reference", 1, sequential_result, nullptr);
                }

                // Each implementation uses its own one-thread reference.
                auto run_parallel_cases = [&](const char* implementation, auto run_kmeans) {
                    BenchmarkResult one_thread_result;
                    if (one_thread_selected || MEASURE_ONE_THREAD_BASELINES) {
                        configure_execution(1, original_mask);
                        one_thread_result = benchmark(implementation, run_kmeans, points, k);
                        record_result(implementation,
                                      one_thread_selected ? "one_thread_reference_and_selected" : "one_thread_reference",
                                      1, one_thread_result, &one_thread_result);
                    }

                    for (int threads : thread_counts) {
                        if (threads == 1) {
                            continue; // Already measured and reused as the reference.
                        }
                        configure_execution(threads, original_mask);
                        const BenchmarkResult selected_result = benchmark(implementation, run_kmeans, points, k);
                        record_result(implementation, "selected_configuration", threads, selected_result,
                                      one_thread_result.times.empty() ? nullptr : &one_thread_result);
                    }
                };

                if (RUN_OPENMP) {
                    auto run_openmp = [&flat_points, n, dimensions](
                        Centroids& centroids, vector<int>& labels, int max_iterations
                    ) {
                        return kmeans_openmp(flat_points, n, dimensions, centroids, labels, max_iterations);
                    };
                    run_parallel_cases("OpenMP AoS", run_openmp);
                }

                if (RUN_OPENMP_SOA) {
                    auto run_soa = [&soa_points, n, dimensions](
                        Centroids& centroids, vector<int>& labels, int max_iterations
                    ) {
                        return kmeans_openmp_soa(soa_points, n, dimensions, centroids, labels, max_iterations);
                    };
                    run_parallel_cases("OpenMP SoA static", run_soa);
                }

                if (RUN_OPENMP_SOA_DYNAMIC) {
                    auto run_soa_dynamic = [&soa_points, n, dimensions](
                        Centroids& centroids, vector<int>& labels, int max_iterations
                    ) {
                        return kmeans_openmp_soa_dynamic(soa_points, n, dimensions, centroids, labels, max_iterations);
                    };
                    run_parallel_cases("OpenMP SoA dynamic", run_soa_dynamic);
                }
            }
        }

        cout << "\nCompleted. CSV files: " << filesystem::absolute(result_directory).string() << "\n";
        return 0;
    } catch (const exception& error) {
        cerr << error.what() << "\n";
        return 1;
    }
}

// Prevent Windows min/max macros from conflicting with C++ functions.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "kmeans_sequential.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

// Change the dataset here before compiling.
constexpr const char* DATASET_PATH =
    "datasets/blobs_N100000_D16.txt";

constexpr int K = 10;
constexpr int MAX_ITERATIONS = 100;
constexpr int WARMUP_RUNS = 2;
constexpr int MEASURED_RUNS = 50;
constexpr bool PIN_TO_CORE = true;
constexpr int LOGICAL_CPU = 2;

static_assert(MEASURED_RUNS > 0,
              "MEASURED_RUNS must be positive.");

static_assert(WARMUP_RUNS >= 0,
              "WARMUP_RUNS cannot be negative.");

static_assert(LOGICAL_CPU >= 0 &&
              LOGICAL_CPU < static_cast<int>(sizeof(DWORD_PTR) * 8),
              "Invalid logical CPU index.");


vector<vector<double>> load_dataset(const char* filename) {
    ifstream file(filename);

    if (!file) {
        throw runtime_error("Cannot open the dataset file.");
    }

    int n;
    int dimensions;

    if (!(file >> n >> dimensions) || n <= 0 || dimensions <= 0) {
        throw runtime_error("Invalid dataset header.");
    }

    vector<vector<double>> points(
        n, vector<double>(dimensions)
    );

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dimensions; d++) {
            if (!(file >> points[i][d])) {
                throw runtime_error("Invalid or incomplete dataset.");
            }
        }
    }

    return points;
}

int main() {
    // Optionally restrict execution to one logical processor.
    if (PIN_TO_CORE) {
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
        // Load the dataset outside the measured region.
        vector<vector<double>> points = load_dataset(DATASET_PATH);

        int n = static_cast<int>(points.size());
        int dimensions = static_cast<int>(points[0].size());

        if (K <= 0 || K > n || MAX_ITERATIONS <= 0) {
            throw runtime_error("Invalid K or iteration limit.");
        }

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

        vector<vector<double>> sums(
            K, vector<double>(dimensions, 0.0)
        );
        vector<int> counts(K, 0);

        vector<double> times(MEASURED_RUNS);
        vector<int> iteration_counts(MEASURED_RUNS);

        cout << "Dataset: " << DATASET_PATH << "\n";
        cout << "Points: " << n << "\n";
        cout << "Dimensions: " << dimensions << "\n";
        cout << "Clusters: " << K << "\n";
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

            int iterations = kmeans(
                points,
                centroids,
                labels,
                sums,
                counts,
                MAX_ITERATIONS
            );

            auto end = chrono::steady_clock::now();

            if (run >= WARMUP_RUNS) {
                int index = run - WARMUP_RUNS;

                times[index] =
                    chrono::duration<double>(end - start).count();

                iteration_counts[index] = iterations;
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

        // Print the centroids from the last run.
        cout << "\nFinal centroids:\n";

        for (int c = 0; c < K; c++) {
            cout << "Centroid " << c << ": ";

            for (int d = 0; d < dimensions; d++) {
                cout << centroids[c][d] << " ";
            }

            cout << "\n";
        }

        return 0;
    } catch (const exception& error) {
        cerr << error.what() << "\n";
        return 1;
    }
}


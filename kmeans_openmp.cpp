#include "kmeans_openmp.h"
#include <omp.h>
#include <cstddef>

int kmeans_openmp(
    const std::vector<double>& points,
    int n,
    int dimensions,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    int max_iterations
) {
    int k = static_cast<int>(centroids.size());

    // Allocate one cluster counter per thread and cluster.
    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_counts(
        max_threads, std::vector<int>(k, 0)
    );

    // Store the coordinate sums for each thread and cluster.
    std::vector<std::vector<std::vector<double>>> thread_sums(
        max_threads,
        std::vector<std::vector<double>>(
            k, std::vector<double>(dimensions, 0.0)
        )
    );

    int iterations = 0;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        bool changed = false;

        // Assign each point to its nearest centroid.
        #pragma omp parallel for default(none) \
            shared(points, centroids, labels, n, k, dimensions) \
            schedule(static) reduction(|| : changed)
        for (int i = 0; i < n; i++) {
            int best_cluster = -1;
            double best_distance = 0.0;

            for (int c = 0; c < k; c++) {
                double distance = 0.0;

                // Compute the squared Euclidean distance.
                for (int d = 0; d < dimensions; d++) {
                    double difference = points[static_cast<std::size_t>(i) * dimensions + d] - centroids[c][d];
                    distance += difference * difference;
                }

                // Ties are resolved by choosing the first centroid.
                if (best_cluster == -1 || distance < best_distance) {
                    best_distance = distance;
                    best_cluster = c;
                }
            }

            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed = true;
            }
        }

        // Reset the partial accumulators for this iteration.
        for (int t = 0; t < max_threads; t++) {
            for (int c = 0; c < k; c++) {
                thread_counts[t][c] = 0;

                for (int d = 0; d < dimensions; d++) {
                    thread_sums[t][c][d] = 0.0;
                }
            }
        }

        // Accumulate point coordinates in the thread's own buffers.
        #pragma omp parallel default(none) \
            shared(points, labels, thread_counts, thread_sums, n, dimensions)
        {
            const int t = omp_get_thread_num();

            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                const int cluster = labels[i];
                thread_counts[t][cluster]++;

                for (int d = 0; d < dimensions; d++) {
                    thread_sums[t][cluster][d] += points[static_cast<std::size_t>(i) * dimensions + d];
                }
            }
        }

        // Merge the partial results and update each centroid.
        for (int c = 0; c < k; c++) {
            int total_count = 0;

            for (int t = 0; t < max_threads; t++) {
                total_count += thread_counts[t][c];
            }

            // Empty clusters keep their previous centroids.
            if (total_count > 0) {
                for (int d = 0; d < dimensions; d++) {
                    double total_sum = 0.0;

                    for (int t = 0; t < max_threads; t++) {
                        total_sum += thread_sums[t][c][d];
                    }

                    centroids[c][d] = total_sum / total_count;
                }
            }
        }

        iterations++;

        // Stop when all assignments remain unchanged.
        if (!changed) {
            break;
        }
    }

    return iterations;
}

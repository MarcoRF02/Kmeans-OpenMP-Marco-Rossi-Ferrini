#include "kmeans_openmp.h"
#include <omp.h>
#include <cstddef>
#include <array>

namespace {
constexpr int POINT_BLOCK_SIZE = 32;
static_assert(POINT_BLOCK_SIZE > 0, "POINT_BLOCK_SIZE must be positive.");
}

int kmeans_openmp(
    const std::vector<double>& points,
    int n,
    int dimensions,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    int max_iterations
) {
    int k = static_cast<int>(centroids.size());
    const int number_of_blocks = n / POINT_BLOCK_SIZE
        + (n % POINT_BLOCK_SIZE != 0);

    // Allocate one cluster counter per thread and cluster.
    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_counts(
        max_threads, std::vector<int>(k, 0)
    );

    // Store partial sums in one contiguous buffer: thread, cluster, coordinate.
    std::vector<double> thread_sums(
        static_cast<std::size_t>(max_threads) * k * dimensions,
        0.0
    );

    int iterations = 0;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        bool changed = false;

        // Reset the partial accumulators for this iteration.
        for (int t = 0; t < max_threads; t++) {
            for (int c = 0; c < k; c++) {
                thread_counts[t][c] = 0;

                for (int d = 0; d < dimensions; d++) {
                    thread_sums[
                        (static_cast<std::size_t>(t) * k + c)
                        * dimensions + d
                    ] = 0.0;
                }
            }
        }

        // Assign points and accumulate coordinates in each thread's buffers.
        #pragma omp parallel default(none) \
            shared(points, centroids, labels, thread_counts, thread_sums, n, k, dimensions, number_of_blocks) \
            reduction(|| : changed)
        {
            const int t = omp_get_thread_num();

            // Each thread owns its temporary arrays.
            std::array<double, POINT_BLOCK_SIZE> distances;
            std::array<double, POINT_BLOCK_SIZE> best_distances;
            std::array<int, POINT_BLOCK_SIZE> best_clusters;

            #pragma omp for schedule(static) nowait
            for (int block = 0; block < number_of_blocks; block++) {
                const int first_point = block * POINT_BLOCK_SIZE;
                const int remaining_points = n - first_point;
                const int block_points = remaining_points < POINT_BLOCK_SIZE
                    ? remaining_points : POINT_BLOCK_SIZE;

                for (int offset = 0; offset < block_points; offset++) {
                    best_clusters[offset] = -1;
                    best_distances[offset] = 0.0;
                }

                for (int c = 0; c < k; c++) {
                    for (int offset = 0; offset < block_points; offset++) {
                        distances[offset] = 0.0;
                    }

                    // Read the same coordinate of consecutive points.
                    for (int d = 0; d < dimensions; d++) {
                        const double centroid_coordinate = centroids[c][d];
                        const std::size_t coordinate_start =
                            static_cast<std::size_t>(first_point)
                            * dimensions + d;

                        for (int offset = 0; offset < block_points; offset++) {
                            double difference = points[
                                coordinate_start
                                + static_cast<std::size_t>(offset) * dimensions
                            ] - centroid_coordinate;

                            distances[offset] += difference * difference;
                        }
                    }

                    // Ties are resolved by choosing the first centroid.
                    for (int offset = 0; offset < block_points; offset++) {
                        if (best_clusters[offset] == -1 ||
                            distances[offset] < best_distances[offset]) {
                            best_distances[offset] = distances[offset];
                            best_clusters[offset] = c;
                        }
                    }
                }

                for (int offset = 0; offset < block_points; offset++) {
                    const int i = first_point + offset;
                    const int best_cluster = best_clusters[offset];

                    if (labels[i] != best_cluster) {
                        labels[i] = best_cluster;
                        changed = true;
                    }

                    // Accumulate every point, even if its label did not change.
                    thread_counts[t][best_cluster]++;

                    for (int d = 0; d < dimensions; d++) {
                        thread_sums[
                            (static_cast<std::size_t>(t) * k + best_cluster)
                            * dimensions + d
                        ] += points[
                            static_cast<std::size_t>(i) * dimensions + d
                        ];
                    }
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
                        total_sum += thread_sums[
                            (static_cast<std::size_t>(t) * k + c)
                            * dimensions + d
                        ];
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
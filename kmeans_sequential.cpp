#include "kmeans_sequential.h"

int kmeans(
    const std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    std::vector<std::vector<double>>& sums,
    std::vector<int>& counts,
    int max_iterations
) {
    int n = static_cast<int>(points.size());
    int dimensions = static_cast<int>(points[0].size());
    int k = static_cast<int>(centroids.size());

    int iterations = 0;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        bool changed = false;

        // Assign each point to its nearest centroid.
        for (int i = 0; i < n; i++) {
            int best_cluster = -1;
            double best_distance = 0.0;

            for (int c = 0; c < k; c++) {
                double distance = 0.0;

                // Compute the squared Euclidean distance.
                for (int d = 0; d < dimensions; d++) {
                    double difference = points[i][d] - centroids[c][d];
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

        // Reset cluster accumulators.
        for (int c = 0; c < k; c++) {
            counts[c] = 0;

            for (int d = 0; d < dimensions; d++) {
                sums[c][d] = 0.0;
            }
        }

        // Accumulate point coordinates for each cluster.
        for (int i = 0; i < n; i++) {
            int cluster = labels[i];
            counts[cluster]++;

            for (int d = 0; d < dimensions; d++) {
                sums[cluster][d] += points[i][d];
            }
        }

        // Update centroids. Empty clusters keep their previous centroids.
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dimensions; d++) {
                    centroids[c][d] = sums[c][d] / counts[c];
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

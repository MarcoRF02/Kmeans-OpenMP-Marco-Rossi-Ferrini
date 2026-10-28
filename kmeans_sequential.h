
#ifndef MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_SEQUENTIAL_H
#define MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_SEQUENTIAL_H

#include <vector>

// Run sequential K-Means using the provided working arrays.
int kmeans(
    const std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    std::vector<std::vector<double>>& sums,
    std::vector<int>& counts,
    int max_iterations
);

#endif //MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_SEQUENTIAL_H

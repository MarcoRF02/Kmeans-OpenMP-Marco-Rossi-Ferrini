#ifndef MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_OPENMP_H
#define MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_OPENMP_H
#include <vector>

int kmeans_openmp(
    const std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    std::vector<std::vector<double>>& sums,
    std::vector<int>& counts,
    int max_iterations
);
#endif //MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_OPENMP_H

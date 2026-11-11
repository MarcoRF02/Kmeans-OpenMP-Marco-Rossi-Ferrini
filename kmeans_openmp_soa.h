#ifndef MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_OPENMP_SOA_H
#define MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_OPENMP_SOA_H

#include <vector>

int kmeans_openmp_soa(
    const std::vector<double>& points,
    int n,
    int dimensions,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    int max_iterations
);

#endif // MARCO_ROSSI_FERRINI_KMEANS_OPENMP_KMEANS_OPENMP_SOA_H
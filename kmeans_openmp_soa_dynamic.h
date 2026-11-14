#ifndef KMEANS_OPENMP_SOA_DYNAMIC_H
#define KMEANS_OPENMP_SOA_DYNAMIC_H

#include <vector>

int kmeans_openmp_soa_dynamic(
    const std::vector<double>& points,
    int n,
    int dimensions,
    std::vector<std::vector<double>>& centroids,
    std::vector<int>& labels,
    int max_iterations
);

#endif // KMEANS_OPENMP_SOA_DYNAMIC_H

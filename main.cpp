#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Versione OpenMP: " << _OPENMP << '\n';

#pragma omp parallel
    {
#pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num()
                      << " di " << omp_get_num_threads()
                      << '\n';
        }
    }

    return 0;
}


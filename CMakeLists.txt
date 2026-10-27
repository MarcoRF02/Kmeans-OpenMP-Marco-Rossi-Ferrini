
cmake_minimum_required(VERSION 4.3)
project(Marco_Rossi_Ferrini_Kmeans_OpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenMP package
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Create the executable
add_executable(Marco_Rossi_Ferrini_Kmeans_OpenMP main.cpp)

# Enable OpenMP for the executable
target_link_libraries(Marco_Rossi_Ferrini_Kmeans_OpenMP
        PRIVATE OpenMP::OpenMP_CXX)

# Explicitly enable OpenMP during linking
target_link_options(Marco_Rossi_Ferrini_Kmeans_OpenMP
        PRIVATE -fopenmp)



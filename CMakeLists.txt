cmake_minimum_required(VERSION 3.16)
project(radixSort LANGUAGES CXX)

# Use modern C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find TBB (Intel oneTBB)
find_package(TBB REQUIRED)

# Add executable
add_executable(radixSort main.cpp)

# Link against TBB
target_link_libraries(radixSort PRIVATE TBB::tbb)

# (Optional) Add warnings and optimizations
target_compile_options(radixSort PRIVATE -Wall -O3)

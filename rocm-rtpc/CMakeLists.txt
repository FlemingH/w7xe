cmake_minimum_required(VERSION 3.21)

project(ROCM_RTPC
    VERSION 1.0.0
    DESCRIPTION "ROCm Real-Time Plasma Computation Platform"
    LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# HIP
find_package(hip REQUIRED)

# GPU architecture (RDNA 4 / gfx1201 for R9700)
if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx1201" CACHE STRING "GPU target architectures")
endif()

# Common include
include_directories(${CMAKE_SOURCE_DIR}/include)

# Compiler flags
add_compile_options(-Wall -Wextra -O3)

# Sub-projects
add_subdirectory(src/common)
add_subdirectory(src/gpu_efit)
add_subdirectory(src/ray_tracing)
add_subdirectory(src/distributed)

# Mark main entry points as HIP so they can link device code
set_source_files_properties(src/gpu_efit/main.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/ray_tracing/main.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(tests/e2e_test.cpp PROPERTIES LANGUAGE HIP)

# Executables
add_executable(gpu_efit_server src/gpu_efit/main.cpp)
target_link_libraries(gpu_efit_server PRIVATE gpu_efit common distributed hip::device)

add_executable(ray_tracing_server src/ray_tracing/main.cpp)
target_link_libraries(ray_tracing_server PRIVATE ray_tracing common distributed hip::device)

# Test executable (single-process end-to-end)
add_executable(e2e_test tests/e2e_test.cpp)
target_link_libraries(e2e_test PRIVATE gpu_efit ray_tracing common distributed hip::device)

# Performance benchmark
set_source_files_properties(tests/benchmark.cpp PROPERTIES LANGUAGE HIP)
add_executable(benchmark tests/benchmark.cpp)
target_link_libraries(benchmark PRIVATE gpu_efit ray_tracing common distributed hip::device)

# RPATH
set_target_properties(gpu_efit_server ray_tracing_server e2e_test benchmark PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib")

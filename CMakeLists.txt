cmake_minimum_required(VERSION 3.24)
project(gds_lab VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(gds_lab
    src/main.cpp
    src/backends.cpp
    src/cuda_pipeline.cu
)

target_include_directories(gds_lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gds_lab PRIVATE CUDA::cudart)
target_compile_definitions(gds_lab PRIVATE _GNU_SOURCE)
set_target_properties(gds_lab PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(gds_lab PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

option(GDSLAB_ENABLE_CUFILE "Build the cuFile backend when libcufile is available" ON)

if(GDSLAB_ENABLE_CUFILE)
    find_path(CUFILE_INCLUDE_DIR
        NAMES cufile.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include
    )
    find_library(CUFILE_LIBRARY
        NAMES cufile
        HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64 /usr/local/cuda/lib
    )
endif()

if(GDSLAB_ENABLE_CUFILE AND CUFILE_INCLUDE_DIR AND CUFILE_LIBRARY)
    message(STATUS "cuFile backend: enabled")
    target_sources(gds_lab PRIVATE src/cufile_backend.cpp)
    target_include_directories(gds_lab PRIVATE ${CUFILE_INCLUDE_DIR})
    target_link_libraries(gds_lab PRIVATE ${CUFILE_LIBRARY})
    target_compile_definitions(gds_lab PRIVATE GDSLAB_HAS_CUFILE=1)
else()
    message(STATUS "cuFile backend: disabled (cufile.h/libcufile not found)")
    target_compile_definitions(gds_lab PRIVATE GDSLAB_HAS_CUFILE=0)
endif()

# avoid policy warning message
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif (POLICY CMP0104)
# main project
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)
project(HERMES C CXX CUDA)
cmake_minimum_required(VERSION 3.18)
cmake_policy(VERSION 3.1)
set(CMAKE_VERBOSE_MAKEFILE ON)
##########################################
##               options                ##
##########################################
option(BUILD_ALL "build all libraries" OFF)
option(BUILD_TESTS "build library unit tests" OFF)
option(BUILD_EXAMPLES "build library examples" OFF)
option(BUILD_SHARED "build shared library" OFF)
option(BUILD_DOCS "build library documentation" OFF)
set(INSTALL_PATH ${BUILD_ROOT} CACHE STRING "include and lib folders path")
##########################################
##            compilation               ##
##########################################
option(BUILD_ALL "build all libraries" OFF)
# cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
# check for specific machine/compiler options.
include(ConfigCompilerFlags)
#find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
##########################################
##           dependencies               ##
##########################################
# include external libraries path
include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext")
include(ponos)
include(catch2)
##########################################
##              source                  ##
##########################################
set(HERMES_HEADERS
        hermes/hermes.h
        hermes/algorithms/cuda_marching_cubes.h
        hermes/colors/cuda_color.h
        hermes/colors/cuda_color_palette.h
        hermes/blas/blas.h
        hermes/blas/vector.h
        hermes/common/cuda.h
        hermes/common/cuda_parallel.h
        hermes/common/cuda_random.h
        hermes/common/defs.h
        hermes/common/index.h
        hermes/common/size.h
        hermes/common/reduce.h
        hermes/geometry/bbox.h
        hermes/geometry/matrix.h
        hermes/geometry/normal.h
        hermes/geometry/numeric.h
        hermes/geometry/transform.h
        hermes/geometry/vector.h
        hermes/geometry/point.h
        hermes/numeric/cuda_field.h
        hermes/numeric/cuda_interpolation.h
        hermes/numeric/vector_grid.h
        hermes/numeric/fd_matrix.h
        hermes/numeric/grid.h
        hermes/numeric/cuda_grid_kernels.h
        hermes/numeric/cuda_numeric.h
        hermes/numeric/cuda_staggered_grid.h
        hermes/storage/array.h
        hermes/storage/cu_array.h
        hermes/storage/cuda_memory_block.h
        hermes/storage/cuda_texture.h
        hermes/storage/cuda_texture.inl
        hermes/storage/cuda_texture_kernels.h
        hermes/storage/cuda_storage_utils.h
        )
set(HERMES_SOURCES
        hermes/algorithms/cuda_marching_cubes.cu
        hermes/numeric/cuda_interpolation.cu
        hermes/storage/cuda_texture_kernels.cu
        )

#enable_language("CUDA")
# list(APPEND CUDA_NVCC_FLAGS -arch compute_35 -use_fast_math –expt-extended-lambda)
add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
# if (MSVC)
#     list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
# endif (MSVC)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#CUDA_COMPILE_PTX(ptx_files ${HERMES_SOURCES} ${HERMES_HEADERS})
add_library(hermes STATIC ${HERMES_SOURCES} ${HERMES_HEADERS})# ${ptx_files})
set_target_properties(hermes PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_STANDARD 17
        CMAKE_CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CMAKE_CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE OFF
        OUTPUT_NAME "hermes"
        FOLDER "HERMES")
target_compile_options(hermes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        -Xcompiler -pg
        --relocatable-device-code=true
        #  -arch=sm_50
        #  -–exp-extended-lambda
        >)
#target_compile_options(hermes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)

target_include_directories(hermes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PONOS_INCLUDES})
target_link_libraries(hermes PUBLIC ${PONOS_LIBRARIES})
add_dependencies(hermes ponos)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hermes
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS hermes
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
##########################################
##              tests                   ##
##########################################
if (BUILD_TESTS)
    add_subdirectory(tests)
endif (BUILD_TESTS)
##########################################
##           documentation              ##
##########################################
if (BUILD_DOCS)
    # TODO
endif (BUILD_DOCS)
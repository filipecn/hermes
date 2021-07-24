##########################################
##                cmake                 ##
##########################################
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
# avoid policy warning message
cmake_policy(SET CMP0104 NEW)
##########################################
##               options                ##
##########################################
option(BUILD_ALL "build all libraries" OFF)
option(BUILD_WITH_CUDA "build with cuda" OFF)
option(BUILD_TESTS "build library unit tests" OFF)
option(BUILD_EXAMPLES "build library examples" OFF)
option(BUILD_SHARED "build shared library" OFF)
option(BUILD_DOCS "build library documentation" OFF)
##########################################
##                project               ##
##########################################
project(HERMES CXX)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)
if (BUILD_WITH_CUDA)
    enable_language(CUDA)
endif (BUILD_WITH_CUDA)
##########################################
##            compilation               ##
##########################################
# cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
# check for specific machine/compiler options.
include(ConfigCompilerFlags)
##########################################
##           dependencies               ##
##########################################
# include external libraries path
include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext")
include(catch2)
##########################################
##              source                  ##
##########################################
set(HERMES_HEADERS
        #        hermes/hermes.h
        #        hermes/algorithms/cuda_marching_cubes.h
        #        hermes/colors/cuda_color.h
        #        hermes/colors/cuda_color_palette.h
        #        hermes/blas/blas.h
        #        hermes/blas/vector.h
        #        hermes/common/cuda.h
        hermes/common/arg_parser.h
        hermes/common/bitmask_operators.h
        hermes/common/cuda_utils.h
        hermes/common/debug.h
        hermes/common/defs.h
        hermes/common/file_system.h
        hermes/common/index.h
        hermes/common/size.h
        hermes/common/str.h
        hermes/common/timer.h
        hermes/data_structures/n_mesh.h
        hermes/data_structures/p_mesh.h
        hermes/data_structures/raw_mesh.h
        #        hermes/common/reduce.h
        hermes/geometry/bbox.h
        hermes/geometry/frustum.h
        hermes/geometry/line.h
        hermes/geometry/matrix.h
        hermes/geometry/normal.h
        hermes/geometry/parametric_surface.h
        hermes/geometry/plane.h
        hermes/geometry/point.h
        hermes/geometry/queries.h
        hermes/geometry/ray.h
        hermes/geometry/segment.h
        hermes/geometry/shape.h
        hermes/geometry/sphere.h
        hermes/geometry/surface.h
        hermes/geometry/transform.h
        hermes/geometry/utils.h
        hermes/geometry/vector.h
        hermes/logging/console_colors.h
        hermes/logging/logging.h
        hermes/logging/memory_dump.h
        hermes/numeric/e_float.h
        hermes/numeric/interpolation.h
        hermes/numeric/interval.h
        #        hermes/numeric/cuda_field.h
        #        hermes/numeric/cuda_interpolation.h
        #        hermes/numeric/numeric.h
        #        hermes/numeric/vector_grid.h
        #        hermes/numeric/fd_matrix.h
        #        hermes/numeric/grid.h
        #        hermes/numeric/cuda_grid_kernels.h
        #        hermes/numeric/cuda_numeric.h
        #        hermes/numeric/cuda_staggered_grid.h
        #        hermes/numeric/math_element.h
        #        hermes/numeric/numeric.h
        hermes/parallel/reduce.h
        #        hermes/random/noise.h
        #        hermes/random/rng.h
        #        hermes/storage/array.h
        #        hermes/storage/array_of_structures.h
        hermes/storage/array.h
        hermes/storage/array_of_structures.h
        hermes/storage/array_of_structures_view.h
        hermes/storage/array_view.h
        hermes/storage/memory_block.h
        hermes/storage/struct_descriptor.h
        #        hermes/storage/device_array.h
        #        hermes/storage/cu_array.h
        #        hermes/storage/cuda_memory_block.h
        #        hermes/storage/cuda_texture.h
        #        hermes/storage/cuda_texture.inl
        #        hermes/storage/cuda_texture_kernels.h
        #        hermes/storage/cuda_storage_utils.h
        )
file(GLOB HERMES_SOURCES
        hermes/common/*.cpp
        hermes/data_structures/*.cpp
        hermes/geometry/*.cpp
        hermes/logging/*.cpp
        #        hermes/numeric/*.cpp
        #        hermes/numeric/e_float.cpp
        hermes/random/*.cpp
        hermes/storage/*.cpp
        )
##########################################
##           hermes target              ##
##########################################
include(SetupHermesTarget)
##########################################
##           install config             ##
##########################################
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
    include(GenerateDocs)
endif (BUILD_DOCS)
##########################################
##           CXX variables              ##
##########################################
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE "ON")
##########################################
##           compiler flags             ##
##########################################
# TODO supporting only GNUCXX for now...
if (CMAKE_COMPILER_IS_GNUCXX)
    set(DEBUG_FLAGS "-g -pg -Wall -Wextra -O0 -fprofile-arcs -ftest-coverage --coverage -fPIC")
    set(RELEASE_FLAGS "-O3 -fPIC")
endif (CMAKE_COMPILER_IS_GNUCXX)
##########################################
##                CUDA                  ##
##########################################
if (BUILD_WITH_CUDA)
    #find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif (BUILD_WITH_CUDA)
##########################################
##               profiles               ##
##########################################
set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)
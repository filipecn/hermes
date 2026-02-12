# ##############################################################################
#                                                                        CMAKE #
# ##############################################################################
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_VERBOSE_MAKEFILE ON)

# ##############################################################################
#                                                                       CHECKS #
# ##############################################################################
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag(-std=c++23 COMPILER_SUPPORTS_CXX23)
#if(NOT COMPILER_SUPPORTS_CXX23)
#  message(FATAL_ERROR "c++23 support required!")
#endif(NOT COMPILER_SUPPORTS_CXX23)

# endianess
if(CMAKE_CXX_BYTE_ORDER)
  if(CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    set(ARCH_IS_BIG_ENDIAN_TARGET 1)
  else()
    set(ARCH_IS_BIG_ENDIAN_TARGET 0)
  endif()
else()
  include(TestBigEndian)
  test_big_endian(BIG_ENDIAN)
  set(ARCH_IS_BIG_ENDIAN_TARGET ${BIG_ENDIAN})
endif()

# ##############################################################################
#                                               CXX variables / compiler flags #
# ##############################################################################
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE "ON")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (CMAKE_COMPILER_IS_GNUCXX)
  set(DEBUG_FLAGS "-g -pg -Wall -Wextra -O0 -fprofile-arcs -ftest-coverage --coverage -fPIC")
  set(RELEASE_FLAGS "-O3 -fPIC")
elseif(MSVC)
  if(MSVC_VERSION GREATER_EQUAL 1914)
    set(DEBUG_FLAGS "/Zc:__cplusplus /Zc:preprocessor /EHsc /MDd")
    set(RELEASE_FLAGS "/Zc:__cplusplus /Zc:preprocessor /EHsc /MD")
  endif()
endif (CMAKE_COMPILER_IS_GNUCXX)

# ##############################################################################
#                                                                         CUDA #
# ##############################################################################
if (HERMES_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 23)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS Off)
  set(CMAKE_CUDA_EXTENSIONS Off)
endif (HERMES_ENABLE_CUDA)

# ##############################################################################
#                                                                     PROFILES #
# ##############################################################################
set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

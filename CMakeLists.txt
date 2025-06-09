# ##############################################################################
#                                                                        CMAKE #
# ##############################################################################
cmake_minimum_required(VERSION 3.12..3.30 FATAL_ERROR)
cmake_policy(SET CMP0177 NEW)

# ##############################################################################
#                                                                      OPTIONS #
# ##############################################################################
option(BUILD_ALL "build all libraries" OFF)
option(BUILD_TESTS "build library unit tests" OFF)
option(BUILD_EXAMPLES "build library examples" OFF)
option(BUILD_SHARED "build shared library" OFF)
option(BUILD_DOCS "build library documentation" OFF)

# ##############################################################################
#                                                                      PROJECT #
# ##############################################################################
project(
  HERMES 
  VERSION 0.01
  DESCRIPTION "Utilities library."
  LANGUAGES C CXX)

cmake_path(SET HERMES_SOURCE_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}")
cmake_path(SET HERMES_EXT_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/ext")
cmake_path(SET HERMES_INCLUDE_SOURCE NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/hermes")
cmake_path(SET HERMES_INSTALL_DEST NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/ext")

# ##############################################################################
#                                                                  COMPILATION #
# ##############################################################################
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(ConfigCompilerFlags)
include(CMakePrintHelpers)
include(ExternalPrograms)

# ##############################################################################
#                                                                 DEPENDENCIES #
# ##############################################################################
include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/ext")
if(BUILD_TESTS)
  include(catch2)
endif(BUILD_TESTS)

# ##############################################################################
#                                                                       SOURCE #
# ##############################################################################
add_subdirectory(hermes)

set(INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
cmake_print_variables(${INSTALL_PATH})

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/hermes
  DESTINATION ${INSTALL_PATH}/include
  FILES_MATCHING
  REGEX "(.*\\.[inl|h])")

install(
  TARGETS hermes
  LIBRARY DESTINATION ${INSTALL_PATH}/lib
  ARCHIVE DESTINATION ${INSTALL_PATH}//lib)

# ##############################################################################
#                                                                         TEST #
# ##############################################################################
if(BUILD_TESTS)
  add_subdirectory(tests)
endif(BUILD_TESTS)

# ##############################################################################
# COMPILATION/RUN SANITY TEST                                                  #
# ##############################################################################
# try_compile(COMPILATION_TEST_RESULT bindir SOURCES source.cpp)
# try_run(RUN_TEST_RESULT bindr SOURCES source.cpp)

# ##############################################################################
#                                                                DOCUMENTATION #
# ##############################################################################
if(BUILD_DOCS)
  include(GenerateDocs)
endif(BUILD_DOCS)

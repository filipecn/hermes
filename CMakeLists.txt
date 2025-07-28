# ##############################################################################
#                                                                        CMAKE #
# ##############################################################################
cmake_minimum_required(VERSION 3.12..3.30 FATAL_ERROR)
cmake_policy(SET CMP0177 NEW)

# ##############################################################################
#                                                                      OPTIONS #
# ##############################################################################
option(HERMES_BUILD_ALL "build all libraries" OFF)
option(HERMES_BUILD_TESTS "build library unit tests" OFF)
option(HERMES_BUILD_EXAMPLES "build library examples" OFF)
option(HERMES_BUILD_SHARED "build shared library" OFF)
option(HERMES_BUILD_DOCS "build library documentation" OFF)
option(HERMES_INCLUDE_TO_STRING "include hermes::to_string methods" ON)

# ##############################################################################
#                                                                      PROJECT #
# ##############################################################################
project(
  HERMES 
  VERSION 0.01
  DESCRIPTION "Utilities library."
  LANGUAGES C CXX)

cmake_path(SET HERMES_PROJECT_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}")
cmake_path(SET HERMES_EXT_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/ext")
cmake_path(SET HERMES_INCLUDE_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}")

# ##############################################################################
#                                                                  COMPILATION #
# ##############################################################################
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(ConfigCompilerFlags)
include(CMakePrintHelpers)
include(ExternalPrograms)

# try_compile(COMPILATION_TEST_RESULT bindir SOURCES source.cpp)
# try_run(RUN_TEST_RESULT bindr SOURCES source.cpp)

# ##############################################################################
#                                                                 DEPENDENCIES #
# ##############################################################################
add_subdirectory(ext)

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
if(HERMES_BUILD_TESTS)
  add_subdirectory(tests)
endif(HERMES_BUILD_TESTS)

# ##############################################################################
#                                                                DOCUMENTATION #
# ##############################################################################
if(HERMES_BUILD_DOCS)
  include(GenerateDocs)
endif(HERMES_BUILD_DOCS)

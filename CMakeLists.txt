# ##############################################################################
#                                                                        CMAKE #
# ##############################################################################
cmake_minimum_required(VERSION 3.12..3.30 FATAL_ERROR)
cmake_policy(SET CMP0177 NEW)

# ##############################################################################
#                                                                      OPTIONS #
# ##############################################################################
option(HERMES_BUILD_ALL "build all libraries" OFF)
option(HERMES_ENABLE_CUDA "build with cuda support" OFF)
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

if(HERMES_ENABLE_CUDA)
  enable_language(CUDA)
endif(HERMES_ENABLE_CUDA)

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

if(HERMES_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
endif(HERMES_ENABLE_CUDA)

function(setup_target target)

  set_target_properties(
    ${target}
    PROPERTIES CXX_STANDART 23
    CXX_STANDARD_REQUIRED YES
    LINKER_LANGUAGE CXX
  )
  # CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-format-style='file';-fix;-p;")

  if(HERMES_ENABLE_CUDA)

    target_compile_definitions(${target} PUBLIC
      ENABLE_CUDA=1
      __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1
    )

    set_target_properties(${target} PROPERTIES
      LINKER_LANGUAGE CUDA
      CUDA_STANDARD 20
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
      CUDA_ARCHITECTURES "89"
    )

    target_compile_options(${target} PRIVATE 
      $<$<COMPILE_LANGUAGE:CUDA>:
      --generate-line-info
      --use_fast_math
      --relocatable-device-code=true
      >)

    target_include_directories(${target}
      PUBLIC
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

  endif(HERMES_ENABLE_CUDA)

  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(
      ${target}
      PUBLIC
      # cuda
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>,-Xcompiler=-Wextra,>"
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>,-Xcompiler=-Wall,>"
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>,-Xcompiler=-Werror,>"
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>,-Xcompiler=-O0,>"
      # CXX
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>,-Wextra,>"
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>,-Wall,>"
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>,-Werror,>"
      "$<IF:$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>,-O0,>"
      # common
      "$<$<CONFIG:Debug>:-g;-pg;-fprofile-arcs;-ftest-coverage;--coverage;-fPIC>"
      "$<$<CONFIG:Release>:-O3;-fPIC;>")
  endif(CMAKE_COMPILER_IS_GNUCXX)



endfunction()

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

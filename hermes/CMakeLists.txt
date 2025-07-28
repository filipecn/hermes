set(HERMES_HEADERS
  ${HERMES_PROJECT_DIR}/hermes/base/flags.h

  ${HERMES_PROJECT_DIR}/hermes/colors/argb_colors.h

  ${HERMES_PROJECT_DIR}/hermes/core/debug.h
  ${HERMES_PROJECT_DIR}/hermes/core/result.h
  ${HERMES_PROJECT_DIR}/hermes/core/types.h

  ${HERMES_PROJECT_DIR}/hermes/io/console_colors.h
  ${HERMES_PROJECT_DIR}/hermes/io/logger.h
  ${HERMES_PROJECT_DIR}/hermes/io/memory_dumper.h

  ${HERMES_PROJECT_DIR}/hermes/numeric/math.h

  ${HERMES_PROJECT_DIR}/hermes/system/os.h
  ${HERMES_PROJECT_DIR}/hermes/system/profile.h
  ${HERMES_PROJECT_DIR}/hermes/system/time.h
  ${HERMES_PROJECT_DIR}/hermes/system/threads.h
)

set(HERMES_SOURCES
  ${HERMES_PROJECT_DIR}/hermes/io/logger.cpp
  ${HERMES_PROJECT_DIR}/hermes/
  ${HERMES_PROJECT_DIR}/hermes/system/os.cpp
  ${HERMES_PROJECT_DIR}/hermes/system/profile.cpp
  ${HERMES_PROJECT_DIR}/hermes/system/time.cpp
  ${HERMES_PROJECT_DIR}/hermes/system/threads.cpp
)

add_library(hermes STATIC ${HERMES_HEADERS} ${HERMES_SOURCES})

target_include_directories(
  hermes PUBLIC "$<BUILD_INTERFACE:${HERMES_INCLUDE_DIR}>"
  "$<INSTALL_INTERFACE:${HERMES_INCLUDE_DIR}>"
)

set_target_properties(
  hermes
  PROPERTIES CXX_STANDART 23
  CXX_STANDARD_REQUIRED YES
  LINKER_LANGUAGE CXX
  OUTPUT_NAME "hermes"
  FOLDER "HERMES")
# CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-format-style='file';-fix;-p;")

if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(
    hermes
    PUBLIC
    "$<$<CONFIG:Debug>:-g;-pg;-Wall;-Werror;-Wextra;-O0;-fprofile-arcs;-ftest-coverage;--coverage;-fPIC>"
    "$<$<CONFIG:Release>:-O3;-fPIC;>")
endif(CMAKE_COMPILER_IS_GNUCXX)

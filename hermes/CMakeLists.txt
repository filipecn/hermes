set(HERMES_HEADERS
  base/flags.h
  base/index.h
  base/size.h

  colors/argb_colors.h
  colors/color.h

  core/debug.h
  core/ref.h
  core/result.h
  core/types.h

  geometry/bounds.h
  geometry/line.h
  geometry/normal.h
  geometry/plane.h
  geometry/point.h
  geometry/queries.h
  geometry/transform.h
  geometry/vector.h

  io/console_colors.h
  io/logger.h
  io/memory_dumper.h

  numeric/interval.h
  numeric/math.h
  numeric/matrix.h

  storage/aos.h
  storage/block.h
  storage/memory.h

  system/os.h
  system/profile.h
  system/time.h
  system/threads.h
)

set(HERMES_SOURCES
  colors/color.cpp

  geometry/line.cpp
  geometry/plane.cpp
  geometry/queries.cpp
  geometry/transform.cpp

  io/logger.cpp

  storage/aos.cpp
  storage/block.cpp
  storage/memory.cpp
  
  system/os.cpp
  system/profile.cpp
  system/time.cpp
  system/threads.cpp
)

add_library(hermes STATIC ${HERMES_HEADERS} ${HERMES_SOURCES})

target_include_directories(
  hermes PUBLIC "$<BUILD_INTERFACE:${HERMES_INCLUDE_DIR}>"
  "$<INSTALL_INTERFACE:${HERMES_INCLUDE_DIR}>"
)

if(HERMES_INCLUDE_TO_STRING)
  target_compile_definitions(hermes PUBLIC HERMES_INCLUDE_TO_STRING)
endif(HERMES_INCLUDE_TO_STRING)

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

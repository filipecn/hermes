set(HERMES_HEADERS
  common/bitmask_operators.h
  common/profile.h
  common/size.h
  common/str.h
  core/types.h
  log/console_colors.h
  log/log.h
  system/file_system.h
  system/threads.h
  system/time.h
)

set(HERMES_SOURCES
  common/profile.cpp
  common/str.cpp
  log/console_colors.cpp
  log/log.cpp
  system/file_system.cpp
  system/threads.cpp
  system/time.cpp
)
  
add_library(hermes STATIC ${HERMES_HEADERS} ${HERMES_SOURCES})

target_include_directories(
  hermes PUBLIC "$<BUILD_INTERFACE:${HERMES_SOURCE_DIR}>"
                "$<INSTALL_INTERFACE:${HERMES_SOURCE_DIR}>"
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

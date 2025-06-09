set(HERMES_HEADERS
  core/types.h
  common/size.h
)

set(HERMES_SOURCES
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

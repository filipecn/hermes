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

  math/math.h
  math/space_filling.h

  numeric/interpolation.h
  numeric/interval.h
  numeric/matrix.h

  sampling/rng.h
  sampling/sampling.h

  storage/aos.h
  storage/block.h
  storage/memory.h

  system/gpu.h
  system/os.h
  system/profile.h
  system/time.h
  system/threads.h
)

set(HERMES_SOURCES 
  io/logger.cpp

  system/profile.cpp
)

set(HERMES_CUDA_SOURCES
  colors/color.cpp

  geometry/line.cpp
  geometry/plane.cpp
  geometry/queries.cpp
  geometry/transform.cpp

  math/space_filling.cpp

  sampling/sampling.cpp

  storage/aos.cpp
  storage/block.cpp
  storage/memory.cpp
  
  system/os.cpp
  system/time.cpp
  system/threads.cpp
)

add_library(hermes STATIC ${HERMES_HEADERS} ${HERMES_SOURCES} ${HERMES_CUDA_SOURCES})

target_include_directories(
  hermes 
  PUBLIC 
  "$<BUILD_INTERFACE:${HERMES_INCLUDE_DIR}>"
  "$<INSTALL_INTERFACE:${HERMES_INCLUDE_DIR}>"
  INTERFACE
  $<INSTALL_INTERFACE:include/hermes>
)

if(HERMES_INCLUDE_DEBUG_TRAITS)
  target_compile_definitions(hermes PUBLIC HERMES_INCLUDE_DEBUG_TRAITS)
endif(HERMES_INCLUDE_DEBUG_TRAITS)

setup_target(hermes)

set_target_properties(hermes PROPERTIES
  OUTPUT_NAME "hermes"
  FOLDER "HERMES")


if(HERMES_ENABLE_CUDA)
 set_source_files_properties(
   ${HERMES_CUDA_SOURCES}
   PROPERTIES LANGUAGE CUDA)
endif(HERMES_ENABLE_CUDA)


include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(BUILD_TESTS)
  include(catch2)
endif(BUILD_TESTS)

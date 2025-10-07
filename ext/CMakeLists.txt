include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(HERMES_BUILD_TESTS)
  include(catch2)
endif(HERMES_BUILD_TESTS)

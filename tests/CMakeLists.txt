add_executable(hermes_tests 
  base_tests.cpp
  core_tests.cpp
  io_tests.cpp
  geometry_tests.cpp
  math_tests.cpp
  numeric_tests.cpp
  storage_tests.cpp
  system_tests.cu
)

target_link_options(hermes_tests INTERFACE --coverage)

add_dependencies(hermes_tests Catch2 hermes)

target_include_directories(hermes_tests PUBLIC 
  ${CATCH2_INCLUDES} 
  ${HERMES_INCLUDE_DIR}
)

setup_target(hermes_tests)

if (HERMES_ENABLE_CUDA)
  set_source_files_properties(
    geometry_tests.cpp
    storage_tests.cpp
    PROPERTIES LANGUAGE CUDA)
endif(HERMES_ENABLE_CUDA)
  
target_link_libraries(hermes_tests PRIVATE 
  hermes
  Catch2::Catch2WithMain
)

if (UNIX)

  target_link_libraries(hermes_tests 
    PRIVATE
    gcov
  )

  add_custom_target(test_hermes
    COMMAND lcov --directory . --zerocounters
    COMMAND hermes_tests DEPENDS hermes_tests
    COMMAND lcov --directory . --capture -output-file cov.info
    COMMAND lcov --remove cov.info '/usr/*' --output-file cov.info
    COMMAND lcov --remove cov.info '*/ext/*' --output-file cov.info
    COMMAND lcov --remove cov.info '*/tests/*' --output-file cov.info
    COMMAND genhtml cov.info --output-directory out
  )

endif (UNIX)


list(APPEND CMAKE_MODULE_PATH ${CATCH2_EXTRAS})
enable_testing()
include(CTest) 
include(Catch)
catch_discover_tests(hermes_tests)

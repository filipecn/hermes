add_executable(hermes_tests 
  common_tests.cpp
)

target_link_options(hermes_tests INTERFACE --coverage)

add_dependencies(hermes_tests Catch2 hermes)

target_include_directories(hermes_tests PUBLIC 
  ${CATCH2_INCLUDES} 
  ${HERMES_SOURCE_DIR}
)
set_target_properties(hermes_tests PROPERTIES LINKER_LANGUAGE CXX)

if (UNIX)

  target_link_libraries(hermes_tests 
    Catch2::Catch2WithMain
    hermes
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

else (UNIX)

    target_link_libraries(hermes_tests hermes)

endif (UNIX)

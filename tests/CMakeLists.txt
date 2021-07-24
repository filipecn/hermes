set(SOURCES
        main.cu
        #        blas_tests.cu
        common_tests.cpp
        data_structures_tests.cpp
        geometry_tests.cpp
        log_tests.cpp
        numeric_tests.cpp
        random_tests.cpp
        storage_tests.cpp
        #        memory_tests.cu
        )

add_executable(hermes_tests ${SOURCES})
target_include_directories(hermes_tests PUBLIC ${CATCH2_INCLUDES} ${CMAKE_SOURCE_DIR})
set_target_properties(hermes_tests PROPERTIES LINKER_LANGUAGE CXX)

if (BUILD_WITH_CUDA)

    set_source_files_properties(common_tests.cpp
            random_tests.cpp
            PROPERTIES LANGUAGE CUDA)

    add_definitions(
            -DENABLE_CUDA=1
            -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)

    target_compile_options(hermes_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            -Xcompiler -pg
            --relocatable-device-code=true
            #    -–extended-lambda
            >)

    set_target_properties(hermes_tests PROPERTIES
            LINKER_LANGUAGE CUDA
            CUDA_STANDARD 17
            CMAKE_CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)

    set_source_files_properties(
            geometry_tests.cpp
            numeric_tests.cpp
            storage_tests.cpp
            PROPERTIES LANGUAGE CUDA)

endif (BUILD_WITH_CUDA)

if (UNIX)

    target_link_libraries(hermes_tests ${PONOS_LIBRARIES} hermes gcov)

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

    target_link_libraries(hermes_tests ponos hermes)

endif (UNIX)

set(SOURCES
        main.cpp
        #        blas_tests.cu
        common_tests.cpp
        geometry_tests.cu
        log_tests.cpp
        numeric_tests.cu
        storage_tests.cu
        random_tests.cpp
        #        memory_tests.cu
        )

add_executable(hermes_tests ${SOURCES})
target_include_directories(hermes_tests PUBLIC ${CATCH2_INCLUDES})
set_target_properties(hermes_tests PROPERTIES LINKER_LANGUAGE CXX)

if (BUILD_WITH_CUDA)
    target_compile_options(hermes_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            #-arch=sm_50
            -Xcompiler -pg
            --relocatable-device-code=true
            #    -–extended-lambda
            >)
    set_target_properties(hermes_tests PROPERTIES
            LINKER_LANGUAGE CUDA
            CMAKE_CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CMAKE_CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            )
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

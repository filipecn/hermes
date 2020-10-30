
enable_language("CUDA")
# list(APPEND CUDA_NVCC_FLAGS -arch compute_35 -use_fast_math –expt-extended-lambda)
add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
# if (MSVC)
#     list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
# endif (MSVC)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#CUDA_COMPILE_PTX(ptx_files ${HERMES_SOURCES} ${HERMES_HEADERS})
add_library(hermes STATIC ${HERMES_SOURCES} ${HERMES_HEADERS})# ${ptx_files})
set_target_properties(hermes PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_STANDARD 11
        CMAKE_CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CMAKE_CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE OFF
        OUTPUT_NAME "hermes"
        FOLDER "HERMES")
target_compile_options(hermes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        -arch=sm_50
        -Xcompiler -pg
        --relocatable-device-code=true
        #  -–exp-extended-lambda
        >)
target_compile_options(hermes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

target_include_directories(hermes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PONOS_SOURCE_DIR})
#target_link_libraries(hermes PUBLIC ponos)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hermes
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS hermes
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )

if (BUILD_SHARED)
    add_library(hermes_shared SHARED ${HERMES_SOURCES} ${HERMES_HEADERS})
    set_target_properties(hermes_shared PROPERTIES
            LINKER_LANGUAGE CUDA
            CUDA_STANDARD 11
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CMAKE_CUDA_SEPARABLE_COMPILATION ON
            OUTPUT_NAME "hermes_shared"
            FOLDER "HERMES_SHARED")
    target_include_directories(hermes_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    install(TARGETS hermes hermes_shared
            LIBRARY DESTINATION ${INSTALL_PATH}/lib
            ARCHIVE DESTINATION ${INSTALL_PATH}//lib
            )
endif (BUILD_SHARED)
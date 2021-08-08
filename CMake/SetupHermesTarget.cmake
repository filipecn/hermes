##########################################
##                 C++                  ##
##########################################
add_library(hermes STATIC ${HERMES_SOURCES} ${HERMES_HEADERS})
set_target_properties(hermes PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hermes PROPERTIES
        OUTPUT_NAME "hermes"
        FOLDER "HERMES")
target_include_directories(hermes PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
##########################################
##                CUDA                  ##
##########################################
if (BUILD_WITH_CUDA)
    add_definitions(
            -DENABLE_CUDA=1
            -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
    set_target_properties(hermes PROPERTIES
            LINKER_LANGUAGE CUDA
            CUDA_STANDARD 17
            CMAKE_CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_RESOLVE_DEVICE_SYMBOLS OFF
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME "hermes"
            FOLDER "HERMES")
    target_compile_options(hermes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            -Xcompiler -pg
            --relocatable-device-code=true
            #            -arch=sm_50
            #  -–exp-extended-lambda
            >)
    target_include_directories(hermes
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            PUBLIC
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            INTERFACE
            $<INSTALL_INTERFACE:include/hermes>
            )

    set_source_files_properties(
            hermes/geometry/transform.cpp
            hermes/common/cuda_utils.cpp
            PROPERTIES LANGUAGE CUDA)
endif (BUILD_WITH_CUDA)
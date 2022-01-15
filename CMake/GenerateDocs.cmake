# check if Doxygen is installed
find_package(Doxygen)
# find_package(Sphinx)
if (DOXYGEN_FOUND)
    set(DOXYGEN_INPUT_DIRS
            ${HERMES_SOURCE_DIR}/hermes/common)
    set(DOXYGEN_EXCLUDE
            ${HERMES_SOURCE_DIR}/hermes/common/parallel.h)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET
            /home/filipecn/Downloads/doxygen-awesome-css/doxygen-awesome.css)
    set(DOXYGEN_HTML_HEADER ${HERMES_SOURCE_DIR/docs/header.html})
    set(DOXYGEN_HTML_FOOTER ${HERMES_SOURCE_DIR/docs/footer.html})

    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    # create doxygen output dir
    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)

else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the documentation")
endif (DOXYGEN_FOUND)
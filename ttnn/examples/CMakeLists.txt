add_executable(example_add)
target_sources(example_add PRIVATE add/add.cpp)
target_link_libraries(example_add PRIVATE TTNN::CPP)

install(
    DIRECTORY
        add
    # DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples
    # FIXME(afuller): Something funky is happening when installing files into /usr/share/doc on a default Docker image.
    #                 Speculation: some dependency for magic doc handling is missing.
    #                 For now keep it out of 'doc'.
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tt-nn/examples"
    COMPONENT ttnn-examples
)

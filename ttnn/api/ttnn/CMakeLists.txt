set(TTNN_API_JIT_FILES tensor/enum_types.hpp)

foreach(JIT_FILE ${TTNN_API_JIT_FILES})
    get_filename_component(JIT_FILE_PATH ${JIT_FILE} DIRECTORY)
    install(
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/${JIT_FILE}
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/api/ttnn/${JIT_FILE_PATH}
        COMPONENT ttnn-runtime
    )
endforeach()

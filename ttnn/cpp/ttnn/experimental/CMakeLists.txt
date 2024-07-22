
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_dnn)

if(WITH_PYTHON_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_lib)
endif()

add_library(ttnn_kernel_lib INTERFACE)
add_library(TTNN::KernelLib ALIAS ttnn_kernel_lib)

file(
    GLOB kernel_lib_jit_api
    CONFIGURE_DEPENDS
    *.hpp
    *.inl
)
list(SORT kernel_lib_jit_api)

set_target_properties(
    ttnn_kernel_lib
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            jit_api
)

target_sources(
    ttnn_kernel_lib
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernel_lib_jit_api}
)

install(
    TARGETS
        ttnn_kernel_lib
    FILE_SET
    jit_api
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/kernel_lib
        COMPONENT ttnn-runtime
)

add_library(ttnn_op_experimental_dit_minimal_binary ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DitMinimalRmBinary ALIAS ttnn_op_experimental_dit_minimal_binary)

target_precompile_headers(ttnn_op_experimental_dit_minimal_binary REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_dit_minimal_binary)

set_target_properties(
    ttnn_op_experimental_dit_minimal_binary
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_dit_minimal_binary
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/dit_minimal_binary_device_operation.cpp
        device/dit_minimal_binary_program_factory.cpp
        dit_minimal_binary.cpp
)

target_include_directories(ttnn_op_experimental_dit_minimal_binary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_dit_minimal_binary
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_dit_minimal_binary
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/dit_minimal_binary
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_dit_minimal_binary LIBRARY COMPONENT tar)

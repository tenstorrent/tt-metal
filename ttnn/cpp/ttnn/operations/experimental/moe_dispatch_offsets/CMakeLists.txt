add_library(ttnn_op_experimental_moe_dispatch_offsets ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MoeDispatchOffsets ALIAS ttnn_op_experimental_moe_dispatch_offsets)

target_precompile_headers(ttnn_op_experimental_moe_dispatch_offsets REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe_dispatch_offsets)

set_target_properties(
    ttnn_op_experimental_moe_dispatch_offsets
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_moe_dispatch_offsets
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
        device/moe_dispatch_offsets_device_operation.cpp
        device/moe_dispatch_offsets_program_factory.cpp
        moe_dispatch_offsets.cpp
)

target_include_directories(ttnn_op_experimental_moe_dispatch_offsets PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_moe_dispatch_offsets
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_moe_dispatch_offsets
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/moe_dispatch_offsets
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_moe_dispatch_offsets LIBRARY COMPONENT tar)

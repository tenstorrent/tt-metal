add_library(ttnn_op_experimental_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Matmul ALIAS ttnn_op_experimental_matmul)

target_precompile_headers(ttnn_op_experimental_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_matmul)

set_target_properties(
    ttnn_op_experimental_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    attn_matmul/device/kernels/*
    group_attn_matmul/device/kernels/*
)

target_sources(
    ttnn_op_experimental_matmul
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
        attn_matmul/attn_matmul.cpp
        attn_matmul/device/attn_matmul_device_operation.cpp
        attn_matmul/device/attn_matmul_program_factory.cpp
        group_attn_matmul/device/group_attn_matmul_device_operation.cpp
        group_attn_matmul/device/group_attn_matmul_program_factory.cpp
        group_attn_matmul/group_attn_matmul.cpp
)

target_include_directories(ttnn_op_experimental_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_matmul
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/matmul
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_matmul LIBRARY COMPONENT tar)

add_library(ttnn_op_core ${LIB_TYPE})
add_library(TTNN::Ops::Core ALIAS ttnn_op_core)

target_precompile_headers(ttnn_op_core REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_core)

target_sources(
    ttnn_op_core
    PRIVATE
        compute_kernel/compute_kernel_config.cpp
        core.cpp
        to_layout/to_layout_op.cpp
        to_dtype/to_dtype_op.cpp
)

target_include_directories(ttnn_op_core PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_core
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_core LIBRARY COMPONENT tar)

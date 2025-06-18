add_library(ttnn_op_experimental_ssm ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::SSM ALIAS ttnn_op_experimental_ssm)

target_precompile_headers(ttnn_op_experimental_ssm REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ssm)

target_sources(
    ttnn_op_experimental_ssm
    PRIVATE
        hc_sum_reduce/device/hc_sum_reduce_op.cpp
        hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp
        hc_sum_reduce/hc_sum_reduce.cpp
        prefix_scan/device/prefix_scan_op.cpp
        prefix_scan/device/prefix_scan_program_factory.cpp
        prefix_scan/prefix_scan.cpp
        repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_op.cpp
        repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_program_factory.cpp
        repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul.cpp
)

target_include_directories(ttnn_op_experimental_ssm PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_ssm
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_ssm LIBRARY COMPONENT tar)

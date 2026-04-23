# Source files for ttnn_op_experimental_ssm.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_SSM_SRCS
    hc_sum_reduce/device/hc_sum_reduce_device_operation.cpp
    hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp
    hc_sum_reduce/hc_sum_reduce.cpp
    prefix_scan/device/prefix_scan_device_operation.cpp
    prefix_scan/device/prefix_scan_program_factory.cpp
    prefix_scan/prefix_scan.cpp
    repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_device_operation.cpp
    repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_program_factory.cpp
    repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul.cpp
)

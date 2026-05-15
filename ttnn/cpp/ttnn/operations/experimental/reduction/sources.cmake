# Source files for ttnn_op_experimental_reduction.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_REDUCTION_SRCS
    fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp
    fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp
    fast_reduce_nc/fast_reduce_nc.cpp
    deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_device_operation.cpp
    deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_program_factory.cpp
    deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.cpp
    integral_image/device/intimg_device_operation.cpp
    integral_image/device/intimg_program_factory.cpp
    integral_image/intimg.cpp
    deepseek_grouped_gate/deepseek_grouped_gate.cpp
    deepseek_grouped_gate/device/deepseek_grouped_gate_device_operation.cpp
    deepseek_grouped_gate/device/deepseek_grouped_gate_program_factory.cpp
)

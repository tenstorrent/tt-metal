# Source files for ttnn_op_experimental_reduction.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_REDUCTION_API_HEADERS
    deepseek_grouped_gate/deepseek_grouped_gate.hpp
    deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.hpp
    fast_reduce_nc/fast_reduce_nc.hpp
    integral_image/intimg.hpp
)

set(TTNN_OP_EXPERIMENTAL_REDUCTION_SRCS
    fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp
    fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp
    fast_reduce_nc/fast_reduce_nc.cpp
    deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_device_operation.cpp
    deepseek_moe_fast_reduce_nc/device/deepseek_moe_fast_reduce_nc_program_factory.cpp
    deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.cpp
    deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_device_operation.cpp
    deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp
    deepseek_moe_fast_reduce_nc_fused/deepseek_moe_fast_reduce_nc_fused.cpp
    integral_image/device/intimg_device_operation.cpp
    integral_image/device/intimg_program_factory.cpp
    integral_image/intimg.cpp
    deepseek_grouped_gate/deepseek_grouped_gate.cpp
    deepseek_grouped_gate/device/deepseek_grouped_gate_device_operation.cpp
    deepseek_grouped_gate/device/deepseek_grouped_gate_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/reduction/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_REDUCTION_NANOBIND_SRCS
    fast_reduce_nc/fast_reduce_nc_nanobind.cpp
    fast_reduce_nc/fast_reduce_nc_nanobind.cpp
    deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc_nanobind.cpp
    deepseek_moe_fast_reduce_nc_fused/deepseek_moe_fast_reduce_nc_fused_nanobind.cpp
    integral_image/intimg_nanobind.cpp
    deepseek_grouped_gate/deepseek_grouped_gate_nanobind.cpp
)

# Source files for ttnn_op_experimental_ssm.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_SSM_API_HEADERS
    hc_sum_reduce/hc_sum_reduce.hpp
    prefix_scan/prefix_scan.hpp
    repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul.hpp
)

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

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/ssm/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_SSM_NANOBIND_SRCS
    hc_sum_reduce/hc_sum_reduce_nanobind.cpp
    prefix_scan/prefix_scan_nanobind.cpp
    repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_nanobind.cpp
)

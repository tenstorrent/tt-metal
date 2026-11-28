# Source files for ttnn_op_experimental_llama31_8b.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_LLAMA31_8B_API_HEADERS
    argmax_decode/argmax_decode.hpp
    argmax_decode/device/argmax_decode_device_operation.hpp
    argmax_decode/device/argmax_decode_table.hpp
    concat_heads_decode/concat_heads_decode.hpp
    concat_heads_decode/device/concat_heads_decode_device_operation.hpp
    concat_heads_decode/device/concat_heads_decode_plan.hpp
    matmul_bh/device/factory/dram_sharded.hpp
    matmul_bh/device/factory/mcast_1d.hpp
    matmul_bh/device/factory/mcast_2d.hpp
    matmul_bh/device/factory/minimal.hpp
    matmul_bh/device/matmul_bh_l1.hpp
    matmul_bh/device/matmul_bh_device_operation.hpp
    matmul_bh/device/matmul_bh_shapes.hpp
    matmul_bh/device/tables/dram_sharded_plan.hpp
    matmul_bh/device/tables/mcast_1d_plan.hpp
    matmul_bh/device/tables/mcast_2d_plan.hpp
    matmul_bh/device/tables/minimal_plan.hpp
    matmul_bh/matmul_bh.hpp
    qkv_heads_decode/device/qkv_heads_decode_device_operation.hpp
    qkv_heads_decode/device/qkv_heads_decode_plan.hpp
    qkv_heads_decode/qkv_heads_decode.hpp
    rms_norm_residual/device/rms_norm_residual_device_operation.hpp
    rms_norm_residual/device/rms_norm_residual_table.hpp
    rms_norm_residual/rms_norm_residual.hpp
    rms_norm_residual_decode/device/rms_norm_residual_decode_device_operation.hpp
    rms_norm_residual_decode/device/rms_norm_residual_decode_table.hpp
    rms_norm_residual_decode/rms_norm_residual_decode.hpp
    rotary_bf8/device/rotary_bf8_device_operation.hpp
    rotary_bf8/device/rotary_bf8_table.hpp
    rotary_bf8/rotary_bf8.hpp
    sdpa_bh/device/sdpa_bh_device_operation.hpp
    sdpa_bh/device/sdpa_bh_table.hpp
    sdpa_bh/sdpa_bh.hpp
)

set(TTNN_OP_EXPERIMENTAL_LLAMA31_8B_SRCS
    argmax_decode/argmax_decode.cpp
    argmax_decode/device/argmax_decode_device_operation.cpp
    argmax_decode/device/argmax_decode_program_factory.cpp
    concat_heads_decode/concat_heads_decode.cpp
    concat_heads_decode/device/concat_heads_decode_device_operation.cpp
    matmul_bh/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp
    matmul_bh/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp
    matmul_bh/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp
    matmul_bh/device/factory/minimal_matmul_program_descriptor.cpp
    matmul_bh/device/matmul_bh_device_operation.cpp
    matmul_bh/matmul_bh.cpp
    qkv_heads_decode/device/qkv_heads_decode_device_operation.cpp
    qkv_heads_decode/qkv_heads_decode.cpp
    rms_norm_residual/device/rms_norm_residual_device_operation.cpp
    rms_norm_residual/device/rms_norm_residual_program_factory.cpp
    rms_norm_residual/rms_norm_residual.cpp
    rms_norm_residual_decode/device/rms_norm_residual_decode_device_operation.cpp
    rms_norm_residual_decode/rms_norm_residual_decode.cpp
    rotary_bf8/device/rotary_bf8_device_operation.cpp
    rotary_bf8/device/rotary_bf8_program_factory.cpp
    rotary_bf8/rotary_bf8.cpp
    sdpa_bh/device/sdpa_bh_device_operation.cpp
    sdpa_bh/device/sdpa_bh_program_factory.cpp
    sdpa_bh/sdpa_bh.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/llama31_8b/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_LLAMA31_8B_NANOBIND_SRCS
    llama31_8b_nanobind.cpp
    argmax_decode/argmax_decode_nanobind.cpp
    concat_heads_decode/concat_heads_decode_nanobind.cpp
    matmul_bh/matmul_bh_nanobind.cpp
    qkv_heads_decode/qkv_heads_decode_nanobind.cpp
    rms_norm_residual/rms_norm_residual_nanobind.cpp
    rms_norm_residual_decode/rms_norm_residual_decode_nanobind.cpp
    rotary_bf8/rotary_bf8_nanobind.cpp
    sdpa_bh/sdpa_bh_nanobind.cpp
)

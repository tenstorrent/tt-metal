// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulConfig {
    uint32_t M_block_size{};
    uint32_t K_block_size{};
    uint32_t N_block_size{};
    uint32_t subblock_h{};
    uint32_t subblock_w{};

    tt::tt_metal::CoreCoord compute_with_storage_grid_size = {0, 0};
};

struct MinimalMatmulParams {
    std::optional<MinimalMatmulConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<float> fused_ternary_scalar;

    DeviceComputeKernelConfig compute_kernel_config;
    int32_t chunks = 1;  // Number of output tensors to split into (default 1 for backward compat)
    int32_t dim = -1;    // Dimension to split along (default -1)

    // Fused SwiGLU: the weight is a tile-pair-interleaved [gate|up] matrix of width 2N.
    // The op emits silu(gate) * up of width N (half the weight width) in a single matmul.
    bool fuse_swiglu = false;

    // Head-split output (chunks > 1): when head_split_rows > 0 each chunk is written as
    // [M / head_split_rows, chunk width / head_split_dim, head_split_rows, head_split_dim] instead of [..., M, width],
    // i.e. rows are users of head_split_rows rows each and the chunk's columns are heads of head_split_dim.
    uint32_t head_split_rows = 0;
    uint32_t head_split_dim = 0;

    // Head-layout activation: the input [users, heads, rows, head_dim] is read as the [users * rows,
    // heads * head_dim] matrix that concatenating its heads gives (row (user, row), column (head, column)).
    bool in0_heads = false;

    // Packed rows (MinimalMatmulInputs::row_map_tensor) with head-split output: the padded user count of the output.
    uint32_t head_split_users = 0;
};

struct MinimalMatmulInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;
    // Second in0 source. AG-fused matmul: this device's local pre-gather slice. Fused concat: the
    // suffix half of in0's K (input_tensor is the prefix half; the weight is stacked [W_prefix; W_suffix]).
    std::optional<Tensor> optional_input_tensor;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<Tensor> fused_ternary_input_a;  // residual/base (broadcast like bias)
    std::optional<Tensor> fused_ternary_input_b;  // gate/multiplier (full MxN shape)

    // Packed-row map, uint32 ROW_MAJOR [..., 4 * packed row tiles]: entry i = {dest, z0, z1, 0} for packed row tile i.
    // Head-split output: row tile i goes to padded row tile dest; padded row tiles [z0, z1) are written as zeros.
    // in0_heads: row tile i reads padded row tile dest of the head-layout activation. dest 0xFFFFFFFF: capacity slack.
    std::optional<Tensor> row_map_tensor;
};

}  // namespace ttnn::experimental::prim

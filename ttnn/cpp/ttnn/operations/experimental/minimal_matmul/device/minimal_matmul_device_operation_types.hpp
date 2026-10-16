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

    // Parallelize M over the grid's x axis and N over y (true) or the reverse (false). Unset: transposed when M > N.
    // Each axis is padded to a multiple of its core count, so the choice sets the padded work per core.
    std::optional<bool> transpose_core_grid = std::nullopt;
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

    // Head-major split output: empty = row-major chunks. Otherwise [S, Wt, off_0, tot_0, off_1, tot_1, ...]: chunk c
    // (N/chunks tiles = (N/chunks)/Wt heads of Wt tiles) is written into a [B, tot_c, S*32, Wt*32] tensor at heads
    // [off_c, off_c + (N/chunks)/Wt), with M = B * S tiles. Requires optional_output_tensors (one per chunk; several
    // chunks may share a tensor, e.g. a fused QKV projection writing Q, Q, ..., K, V).
    std::vector<uint32_t> head_major;
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

    // Preallocated outputs, one per chunk (required with head_major); empty: the op allocates them.
    std::vector<Tensor> optional_output_tensors;
};

}  // namespace ttnn::experimental::prim

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

    // Grid orientation of minimal_matmul, minimal_matmul_split and dit_minimal_matmul_addcmul_fused:
    // transposed puts M on the cores' x axis and N on y. Unset, the op transposes when M > N, so a matmul
    // whose rows cross N compiles a second program. Setting it keeps one program for every row count.
    // The fused CCL matmuls choose their own orientation and reject a set value.
    std::optional<bool> transpose_core_grid = std::nullopt;
};

// The grid orientation a call runs with: the config's, or transposed when the output is taller than wide.
inline bool resolve_transpose_core_grid(const std::optional<MinimalMatmulConfig>& config, uint32_t M, uint32_t N) {
    if (config.has_value() && config->transpose_core_grid.has_value()) {
        return *config->transpose_core_grid;
    }
    return M > N;
}

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
};

}  // namespace ttnn::experimental::prim

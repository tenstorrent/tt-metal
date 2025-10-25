// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_device_operation.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "minimal_matmul_program_factory.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::minimal_matmul {

void MinimalMatmulOp::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "minimal_matmul expects exactly 2 inputs: activation and weight");

    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const bool has_bias = (optional_input_tensors.size() == 1) && optional_input_tensors.at(0).has_value();
    const Tensor* bias_ptr = has_bias ? &optional_input_tensors.at(0).value() : nullptr;

    // Basic device/storage checks
    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "minimal_matmul operands must be on device");
    TT_FATAL(act_tensor.device() == weight_tensor.device(), "minimal_matmul inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "minimal_matmul inputs must be allocated in device buffers");
    if (has_bias) {
        const auto& bias_tensor = *bias_ptr;
        TT_FATAL(bias_tensor.storage_type() == StorageType::DEVICE, "minimal_matmul bias must be on device");
        TT_FATAL(bias_tensor.device() == act_tensor.device(), "minimal_matmul bias must be on the same device");
        TT_FATAL(bias_tensor.buffer() != nullptr, "minimal_matmul bias must be allocated in a device buffer");
    }

    // Layout requirements: all inputs must be TILE layout
    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "minimal_matmul requires TILE layout for activation and weight");
    if (has_bias) {
        TT_FATAL(bias_ptr->layout() == Layout::TILE, "minimal_matmul requires TILE layout for bias");
    }

    // DType constraints: support BFLOAT16 and BFLOAT8_B; inputs must match
    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "minimal_matmul supports only BFLOAT16, BFLOAT8_B, and BFLOAT4_B for inputs");

    // Bias dtype constraint, if present
    if (has_bias) {
        TT_FATAL(
            dtype_supported(bias_ptr->dtype()),
            "minimal_matmul supports only BFLOAT16, BFLOAT8_B, and BFLOAT4_B for bias");
    }

    // Shape constraints
    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "minimal_matmul expects rank >= 2 tensors");

    // No batching: all leading dims (before last two) must be 1
    for (int i = 0; i < static_cast<int>(a_logical.rank()) - 2; ++i) {
        TT_FATAL(a_logical[i] == 1, "minimal_matmul activation must have 1 in all dims < -2");
    }
    for (int i = 0; i < static_cast<int>(w_logical.rank()) - 2; ++i) {
        TT_FATAL(w_logical[i] == 1, "minimal_matmul weight must have 1 in all dims < -2");
    }

    const uint32_t M = a_logical[-2];
    const uint32_t K = a_logical[-1];
    const uint32_t K_w = w_logical[-2];
    const uint32_t N = w_logical[-1];

    TT_FATAL(K == K_w, "minimal_matmul inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(M > 0 && K > 0 && N > 0, "minimal_matmul dimensions must be positive");

    if (has_bias) {
        const auto& b_logical = bias_ptr->logical_shape();
        TT_FATAL(b_logical.rank() >= 1, "minimal_matmul bias must have rank >= 1");
        // All dims except the last must be 1 (i.e., shape is [..., 1, N])
        for (int i = 0; i < static_cast<int>(b_logical.rank()) - 1; ++i) {
            TT_FATAL(b_logical[i] == 1, "minimal_matmul bias must be 1 in all dims except the last");
        }
        TT_FATAL(b_logical[-1] == N, "minimal_matmul bias last dimension must equal N ({}), got {}", N, b_logical[-1]);
    }

    // Tile alignment checks (implicitly guaranteed by TILE layout, but assert inner two dims are tile-aligned)
    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "minimal_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0,
        "minimal_matmul weight must be tile-aligned");
    if (has_bias) {
        const auto& b_padded = bias_ptr->padded_shape();
        TT_FATAL(b_padded[-1] % TILE_WIDTH == 0, "minimal_matmul bias last dimension must be tile-aligned");
    }

    // Config constraints
    if (config.has_value()) {
        const auto& cfg = config.value();
        TT_FATAL(cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0, "Block sizes must be > 0");
        TT_FATAL(cfg.subblock_h > 0 && cfg.subblock_w > 0, "Subblock sizes must be > 0");
        TT_FATAL(
            (cfg.M_block_size % cfg.subblock_h) == 0,
            "M_block_size ({}) must be divisible by subblock_h ({})",
            cfg.M_block_size,
            cfg.subblock_h);
        TT_FATAL(
            (cfg.N_block_size % cfg.subblock_w) == 0,
            "N_block_size ({}) must be divisible by subblock_w ({})",
            cfg.N_block_size,
            cfg.subblock_w);

        // Grid must be at least 1x1
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be >= 2x2");

        // Additional grid checks are performed when creating the program
        auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "compute_with_storage_grid_size must be <= device grid size");

        const uint32_t max_dest_volume = get_dest_reg_count(this->compute_kernel_config);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");
    }
}

std::vector<TensorSpec> MinimalMatmulOp::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& in0_input_tensor = input_tensors.at(0);
    const auto& in1_input_tensor = input_tensors.at(1);
    const auto& in0_input_tensor_shape = in0_input_tensor.logical_shape();
    const auto& in1_input_tensor_shape = in1_input_tensor.logical_shape();
    uint32_t N = in1_input_tensor_shape[-1];

    ttnn::Shape output_shape(in0_input_tensor_shape);
    output_shape[-1] = N;

    const auto& memory_config = this->output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = in0_input_tensor.dtype();

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks MinimalMatmulOp::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const auto& bias_tensor = optional_input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    return detail::minimal_matmul_factory(
        act_tensor,
        weight_tensor,
        bias_tensor,
        this->fused_activation,
        this->config,
        output_tensor,
        compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::minimal_matmul

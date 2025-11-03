// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::experimental::ccl {

void FusedRMSNormPostAllGather::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors");
    auto& a = input_tensors.at(0);
    auto& stats = input_tensors.at(1);

    // Helper lambda to assert tensor properties: tilized, bfloat16, on device, allocated
    auto check_tile_bf16_device_alloc = [](const Tensor& tensor, const std::string& name) {
        TT_FATAL(tensor.layout() == Layout::TILE, "{} tensor must have TILE layout, got: {}", name, tensor.layout());
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} tensor must be BFLOAT16, got: {}", name, tensor.dtype());
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} tensor must be on device!", name);
        TT_FATAL(tensor.buffer() != nullptr, "{} tensor must be allocated in buffers on device!", name);
    };

    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
        const auto& tensor = input_tensors[idx];
        check_tile_bf16_device_alloc(tensor, "Input tensor " + std::to_string(idx));
    }

    // stats has 1 tile columns per device
    TT_FATAL(
        stats.padded_shape()[-1] % TILE_WIDTH == 0,
        "Stats inner dimension must be divisible by TILE_WIDTH (32), got: {}",
        stats.padded_shape()[-1]);
    // All other dims must match
    TT_FATAL(
        stats.padded_shape().size() == a.padded_shape().size(), "Stats and input must have same number of dimensions");
    for (int i = 0; i < a.padded_shape().size() - 1; i++) {
        TT_FATAL(
            stats.padded_shape()[i] == a.padded_shape()[i],
            "Stats and input dim{} must match, got stats: {} vs input: {}",
            i,
            stats.padded_shape()[i],
            a.padded_shape()[i]);
    }

    TT_FATAL(this->num_heads > 0, "Number of heads must be greater than 0, got: {}", this->num_heads);
    TT_FATAL(
        a.padded_shape()[-1] % this->num_heads == 0,
        "Input last dimension must be divisible by number of heads, got hidden_dim: {} vs num_heads: {}",
        a.padded_shape()[-1],
        this->num_heads);

    TT_FATAL(
        a.logical_shape()[-1] == a.padded_shape()[-1],
        "Input last dimension must be the same as padded last dimension, got logical_dim: {} vs padded_dim: {}",
        a.logical_shape()[-1],
        a.padded_shape()[-1]);

    TT_FATAL(a.logical_shape().rank() == 4, "Input must have rank 4, got: {}", a.logical_shape().rank());
    // Expected input shape: [batch, 1, sequence_length, hidden_dim]
    TT_FATAL(a.logical_shape()[1] == 1, "Input dim 1 must be 1, got: {}", a.logical_shape()[1]);
    TT_FATAL(a.logical_shape()[0] == 1, "Expecting input batch dimension to be 1, got: {}", a.logical_shape()[0]);

    TT_FATAL(optional_input_tensors.size() == 4, "Must have 4 optional input tensors");
    if (optional_input_tensors.at(0).has_value()) {
        // Gamma is given
        auto& weight = optional_input_tensors.at(0).value();
        check_tile_bf16_device_alloc(weight, "Weight");

        TT_FATAL(
            weight.padded_shape().size() == 2,
            "Weight tensor must have 2 dimensions, got: {}",
            weight.padded_shape().size());
        TT_FATAL(
            weight.padded_shape()[-1] == a.padded_shape()[-1],
            "Weight tensor must have same last dimension as input, got: {} vs {}",
            weight.padded_shape()[-1],
            a.padded_shape()[-1]);
        TT_FATAL(
            weight.logical_shape()[0] == 1,
            "Weight tensor must have batch dimension of 1, got: {}",
            weight.logical_shape()[0]);
    }

    if (optional_input_tensors.at(1).has_value()) {
        // ROPE fusion is enabled
        TT_FATAL(optional_input_tensors.at(2).has_value(), "Rope cos tensor is required when ROPE fusion is enabled");
        TT_FATAL(optional_input_tensors.at(3).has_value(), "Rope sin tensor is required when ROPE fusion is enabled");

        auto& transformation_mat = optional_input_tensors.at(1).value();
        auto& rope_cos = optional_input_tensors.at(2).value();
        auto& rope_sin = optional_input_tensors.at(3).value();

        check_tile_bf16_device_alloc(transformation_mat, "Transformation_mat");
        check_tile_bf16_device_alloc(rope_cos, "Rope cos");
        check_tile_bf16_device_alloc(rope_sin, "Rope sin");

        // Ensure transformation_mat has 4 dimensions: [1, 1, 32, 32]
        TT_FATAL(
            transformation_mat.padded_shape().size() == 4,
            "Transformation_mat must have 4 dimensions, got: {}",
            transformation_mat.padded_shape().size());
        TT_FATAL(
            transformation_mat.padded_shape()[0] == 1 && transformation_mat.padded_shape()[1] == 1 &&
                transformation_mat.padded_shape()[2] == 32 && transformation_mat.padded_shape()[3] == 32,
            "Transformation_mat must have shape [1, 1, 32, 32], got: [{} {} {} {}]",
            transformation_mat.padded_shape()[0],
            transformation_mat.padded_shape()[1],
            transformation_mat.padded_shape()[2],
            transformation_mat.padded_shape()[3]);

        // Ensure rope_cos and rope_sin have 4 dimensions: [1, 1, a.padded_shape()[2], head_dim]
        auto seq_len = a.padded_shape()[2];
        auto head_dim = a.padded_shape()[3] / this->num_heads;
        for (const auto& rope_tensor : {std::cref(rope_cos), std::cref(rope_sin)}) {
            TT_FATAL(
                rope_tensor.get().padded_shape().size() == 4,
                "Rope tensor must have 4 dimensions, got: {}",
                rope_tensor.get().padded_shape().size());
            TT_FATAL(
                rope_tensor.get().padded_shape()[0] == 1 && rope_tensor.get().padded_shape()[1] == 1 &&
                    rope_tensor.get().padded_shape()[2] == seq_len && rope_tensor.get().padded_shape()[3] == head_dim,
                "Rope tensor must have shape [1, 1, {}, {}], got: [{} {} {} {}]",
                seq_len,
                head_dim,
                rope_tensor.get().padded_shape()[0],
                rope_tensor.get().padded_shape()[1],
                rope_tensor.get().padded_shape()[2],
                rope_tensor.get().padded_shape()[3]);
        }
    }
}

std::vector<TensorSpec> FusedRMSNormPostAllGather::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensor.logical_shape();
    output_shape[1] = this->num_heads;
    output_shape[3] /= this->num_heads;

    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            this->dtype.value_or(input_tensor.dtype()),
            tt::tt_metal::PageConfig(Layout::TILE),
            input_tensor.memory_config()))};
}

tt::tt_metal::operation::ProgramWithCallbacks FusedRMSNormPostAllGather::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& stats = input_tensors.at(1);
    const auto& weight = optional_input_tensors.at(0);
    const auto& transformation_mat = optional_input_tensors.at(1);
    const auto& rope_cos = optional_input_tensors.at(2);
    const auto& rope_sin = optional_input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    return fused_rmsnorm_post_allgather_multi_core(
        a,
        stats,
        output_tensor,
        weight,
        transformation_mat,
        rope_cos,
        rope_sin,
        this->eps,
        this->num_heads,
        this->compute_kernel_config);
}
}  // namespace ttnn::operations::experimental::ccl

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rmsnorm_post_all_gather_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

FusedRMSNormPostAllGatherDeviceOperation::program_factory_t
FusedRMSNormPostAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return FusedRMSNormPostAllGatherProgramFactory{};
}

void FusedRMSNormPostAllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void FusedRMSNormPostAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor;
    const auto& stats = tensor_args.stats_tensor;
    const auto& weight_opt = tensor_args.weight;
    const auto& transformation_mat_opt = tensor_args.transformation_mat;
    const auto& rope_cos_opt = tensor_args.rope_cos;
    const auto& rope_sin_opt = tensor_args.rope_sin;

    // Helper lambda to assert tensor properties: tilized, bfloat16, on device, allocated
    auto check_tile_bf16_device_alloc = [](const Tensor& tensor, const std::string& name) {
        TT_FATAL(tensor.layout() == Layout::TILE, "{} tensor must have TILE layout, got: {}", name, tensor.layout());
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} tensor must be BFLOAT16, got: {}", name, tensor.dtype());
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} tensor must be on device!", name);
        TT_FATAL(tensor.buffer() != nullptr, "{} tensor must be allocated in buffers on device!", name);
    };

    check_tile_bf16_device_alloc(a, "Input tensor 0");
    check_tile_bf16_device_alloc(stats, "Input tensor 1");

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

    TT_FATAL(args.num_heads > 0, "Number of heads must be greater than 0, got: {}", args.num_heads);
    TT_FATAL(
        a.padded_shape()[-1] % args.num_heads == 0,
        "Input last dimension must be divisible by number of heads, got hidden_dim: {} vs num_heads: {}",
        a.padded_shape()[-1],
        args.num_heads);

    TT_FATAL(
        a.logical_shape()[-1] == a.padded_shape()[-1],
        "Input last dimension must be the same as padded last dimension, got logical_dim: {} vs padded_dim: {}",
        a.logical_shape()[-1],
        a.padded_shape()[-1]);

    TT_FATAL(a.logical_shape().rank() == 4, "Input must have rank 4, got: {}", a.logical_shape().rank());
    // Expected input shape: [batch, 1, sequence_length, hidden_dim]
    TT_FATAL(a.logical_shape()[1] == 1, "Input dim 1 must be 1, got: {}", a.logical_shape()[1]);
    TT_FATAL(a.logical_shape()[0] == 1, "Expecting input batch dimension to be 1, got: {}", a.logical_shape()[0]);

    if (weight_opt.has_value()) {
        // Gamma is given
        const auto& weight = weight_opt.value();
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

    if (transformation_mat_opt.has_value()) {
        // ROPE fusion is enabled
        TT_FATAL(rope_cos_opt.has_value(), "Rope cos tensor is required when ROPE fusion is enabled");
        TT_FATAL(rope_sin_opt.has_value(), "Rope sin tensor is required when ROPE fusion is enabled");

        const auto& transformation_mat = transformation_mat_opt.value();
        const auto& rope_cos = rope_cos_opt.value();
        const auto& rope_sin = rope_sin_opt.value();

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
        auto head_dim = a.padded_shape()[3] / args.num_heads;
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

TensorSpec FusedRMSNormPostAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();
    output_shape[1] = args.num_heads;
    output_shape[3] /= args.num_heads;

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            args.dtype.value_or(input_tensor.dtype()), tt::tt_metal::PageConfig(Layout::TILE), args.memory_config));
}

Tensor FusedRMSNormPostAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_rmsnorm_post_all_gather(
    const Tensor& input_tensor,
    const Tensor& stats_tensor,
    float eps,
    uint32_t num_heads,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype) {
    using OperationType = ttnn::experimental::prim::FusedRMSNormPostAllGatherDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .eps = eps,
        .num_heads = num_heads,
        .memory_config = memory_config,
        .compute_kernel_config = compute_kernel_config,
        .dtype = dtype,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .stats_tensor = stats_tensor,
        .weight = weight.has_value() ? std::optional<Tensor>(weight.value()) : std::nullopt,
        .transformation_mat =
            transformation_mat.has_value() ? std::optional<Tensor>(transformation_mat.value()) : std::nullopt,
        .rope_cos = rope_cos.has_value() ? std::optional<Tensor>(rope_cos.value()) : std::nullopt,
        .rope_sin = rope_sin.has_value() ? std::optional<Tensor>(rope_sin.value()) : std::nullopt,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

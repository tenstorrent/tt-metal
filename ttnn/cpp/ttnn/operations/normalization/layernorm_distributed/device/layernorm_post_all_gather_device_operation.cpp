// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

LayerNormPostAllGatherDeviceOperation::program_factory_t LayerNormPostAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    // Check if Welford algorithm is requested (only for layernorm)
    if (std::holds_alternative<LayerNormDefaultProgramConfig>(args.program_config)) {
        const auto& program_config = std::get<LayerNormDefaultProgramConfig>(args.program_config);
        if (program_config.use_welford) {
            return LayerNormPostAllGatherWelfordProgramFactory{};
        }
    }

    // Default to normal program factory
    return LayerNormPostAllGatherProgramFactory{};
}

void LayerNormPostAllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LayerNormPostAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& stats = tensor_args.stats;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16 || a.dtype() == DataType::BFLOAT8_B,
        "Input tensor must be BFLOAT16 or BFLOAT8_B, got: {}",
        a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    TT_FATAL(stats.layout() == Layout::TILE, "Stats tensor must have TILE layout, got: {}", stats.layout());
    TT_FATAL(
        stats.dtype() == DataType::BFLOAT16 || stats.dtype() == DataType::BFLOAT8_B,
        "Stats tensor must be BFLOAT16 or BFLOAT8_B, got: {}",
        stats.dtype());
    TT_FATAL(stats.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(stats.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    // stats has 2 or 1 tile columns per device if layernorm or rmsnorm
    TT_FATAL(
        stats.padded_shape()[-1] % TILE_WIDTH == 0,
        "Stats inner dimension must be divisible by TILE_WIDTH (32), got: {}",
        stats.padded_shape()[-1]);
    TT_FATAL(
        stats.padded_shape()[0] == a.padded_shape()[0],
        "Stats and input batch sizes must match, got stats: {} vs input: {}",
        stats.padded_shape()[0],
        a.padded_shape()[0]);
    TT_FATAL(
        stats.padded_shape()[1] == a.padded_shape()[1],
        "Stats and input dim1 must match, got stats: {} vs input: {}",
        stats.padded_shape()[1],
        a.padded_shape()[1]);
    TT_FATAL(
        stats.padded_shape()[2] == a.padded_shape()[2],
        "Stats and input dim2 must match, got stats: {} vs input: {}",
        stats.padded_shape()[2],
        a.padded_shape()[2]);

    if (gamma.has_value()) {
        const auto& gamma_tensor = gamma.value();

        if (gamma_tensor.layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma_tensor.padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma_tensor.padded_shape()[-1]);
            TT_FATAL(
                gamma_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma_tensor.device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma_tensor.padded_shape()[-2] == TILE_HEIGHT,
                "Gamma tensor height must be TILE_HEIGHT (32), got: {}",
                gamma_tensor.padded_shape()[-2]);
        } else {
            TT_FATAL(
                gamma_tensor.layout() == Layout::ROW_MAJOR,
                "Gamma tensor must have ROW_MAJOR layout, got: {}",
                gamma_tensor.layout());
            TT_FATAL(
                (gamma_tensor.padded_shape()[-1] == TILE_WIDTH &&
                 gamma_tensor.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Gamma tensor dimensions must align with input tensor. Got gamma padded shape: {}, physical volume: "
                "{}, input padded shape: {}, TILE_WIDTH: {}",
                gamma_tensor.padded_shape()[-1],
                gamma_tensor.physical_volume(),
                a.padded_shape()[-1],
                TILE_WIDTH);
            TT_FATAL(
                gamma_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma_tensor.device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma_tensor.dtype() == DataType::BFLOAT16,
                "Gamma tensor must be BFLOAT16, got: {}",
                gamma_tensor.dtype());
        }
        const bool is_layernorm = args.norm_type == LayerNormDistributedType::LAYERNORM;
        const bool has_beta = beta.has_value();
        TT_FATAL(is_layernorm == has_beta, "Beta tensor must be present if and only if using layernorm (vs rmsnorm)");

        if (beta.has_value()) {
            const auto& beta_tensor = beta.value();
            if (beta_tensor.layout() == Layout::TILE) {
                TT_FATAL(
                    a.padded_shape()[-1] == beta_tensor.padded_shape()[-1],
                    "Input and beta inner dimensions must match, got input: {} vs beta: {}",
                    a.padded_shape()[-1],
                    beta_tensor.padded_shape()[-1]);
                TT_FATAL(
                    beta_tensor.buffer() != nullptr,
                    "Operands to layernorm need to be allocated in buffers on device!");
                TT_FATAL(a.device() == beta_tensor.device(), "Input and beta tensors must be on same device");
                TT_FATAL(
                    beta_tensor.padded_shape()[-2] == TILE_HEIGHT,
                    "Beta tensor height must be TILE_HEIGHT (32), got: {}",
                    beta_tensor.padded_shape()[-2]);
            } else {
                TT_FATAL(
                    beta_tensor.layout() == Layout::ROW_MAJOR,
                    "Beta tensor must have ROW_MAJOR layout, got: {}",
                    beta_tensor.layout());
                TT_FATAL(
                    (beta_tensor.padded_shape()[-1] == TILE_WIDTH &&
                     beta_tensor.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                    "Beta tensor dimensions must align with input tensor. Got beta padded shape: {}, physical volume: "
                    "{}, input padded shape: {}, TILE_WIDTH: {}",
                    beta_tensor.padded_shape()[-1],
                    beta_tensor.physical_volume(),
                    a.padded_shape()[-1],
                    TILE_WIDTH);
                TT_FATAL(
                    beta_tensor.buffer() != nullptr,
                    "Operands to layernorm need to be allocated in buffers on device!");
                TT_FATAL(a.device() == beta_tensor.device(), "Input and beta tensors must be on same device");
                TT_FATAL(
                    beta_tensor.dtype() == DataType::BFLOAT16,
                    "Beta tensor must be BFLOAT16, got: {}",
                    beta_tensor.dtype());
            }
        }
    }

    // Additional validation for Welford - it doesn't support rmsnorm
    if (std::holds_alternative<LayerNormDefaultProgramConfig>(args.program_config)) {
        const auto& program_config = std::get<LayerNormDefaultProgramConfig>(args.program_config);
        TT_FATAL(
            !(program_config.use_welford && args.norm_type == LayerNormDistributedType::RMSNORM),
            "RMS norm is not compatible with Welford algorithm. Please disable use_welford flag.");
    }
}

LayerNormPostAllGatherDeviceOperation::spec_return_value_t LayerNormPostAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(args.dtype.value_or(input_tensor.dtype()), PageConfig(Layout::TILE), args.memory_config));
}

LayerNormPostAllGatherDeviceOperation::tensor_return_value_t
LayerNormPostAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::prim

namespace ttnn::prim {

Tensor layer_norm_post_all_gather(
    const Tensor& input,
    const Tensor& stats,
    LayerNormDistributedType norm_type,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype,
    const std::optional<bool>& use_2d_core_grid,
    const LayerNormProgramConfig& program_config) {
    using OperationType = LayerNormPostAllGatherDeviceOperation;
    return ttnn::device_operation::detail::launch<OperationType>(
        OperationType::operation_attributes_t{
            .norm_type = norm_type,
            .eps = eps,
            .memory_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .dtype = dtype,
            .use_2d_core_grid = use_2d_core_grid,
            .program_config = program_config,
        },
        OperationType::tensor_args_t{
            .input = input,
            .stats = stats,
            .gamma = gamma.has_value() ? std::make_optional(gamma.value()) : std::nullopt,
            .beta = beta.has_value() ? std::make_optional(beta.value()) : std::nullopt});
}

}  // namespace ttnn::prim

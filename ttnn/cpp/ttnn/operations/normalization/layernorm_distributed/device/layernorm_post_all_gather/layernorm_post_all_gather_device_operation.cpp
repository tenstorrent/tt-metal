// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_device_operation.hpp"
#include "ttnn/api/ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/constants.hpp>

#include <string_view>

using namespace tt::constants;

namespace ttnn::operations::normalization::layernorm_post_all_gather {

LayerNormPostAllGatherDeviceOperation::program_factory_t LayerNormPostAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (args.use_2d_core_grid.has_value() && args.use_2d_core_grid.value()) {
        return program::LayerNormPostAllGather2DProgramFactory{};
    }
    return program::LayerNormPostAllGatherProgramFactory{};
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

    auto validate_input_tensor = [](const Tensor& t, std::string_view name) {
        TT_FATAL(t.layout() == Layout::TILE, "{} tensor must have TILE layout, got: {}", name, t.layout());
        TT_FATAL(
            t.dtype() == DataType::BFLOAT16 || t.dtype() == DataType::BFLOAT8_B,
            "{} tensor must be BFLOAT16 or BFLOAT8_B, got: {}",
            name,
            t.dtype());
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} tensor must be on device!", name);
        TT_FATAL(t.buffer() != nullptr, "{} tensor must be allocated in buffers on device!", name);
    };

    validate_input_tensor(a, "Input");
    validate_input_tensor(stats, "Stats");

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
    // TODO: How to check if number of tile columns is correct? Would have to know # of devices and is_rmsnorm

    if (gamma.has_value()) {
        const auto& gamma_tensor = gamma.value();

        TT_FATAL(
            gamma_tensor.layout() == Layout::ROW_MAJOR,
            "Gamma tensor must have ROW_MAJOR layout (only packed RM supported), got: {}",
            gamma_tensor.layout());
        if (gamma_tensor.layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma.value().padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma.value().padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().padded_shape()[-2] == TILE_HEIGHT,
                "Gamma tensor height must be TILE_HEIGHT (32), got: {}",
                gamma.value().padded_shape()[-2]);
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
            TT_FATAL(
                gamma_tensor.layout() == beta_tensor.layout(),
                "Gamma and beta must have the same layout, got gamma: {} vs beta: {}",
                gamma_tensor.layout(),
                beta_tensor.layout());
            TT_FATAL(
                beta_tensor.layout() == Layout::ROW_MAJOR,
                "Beta tensor must have ROW_MAJOR layout, got: {}",
                beta_tensor.layout());
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
                    beta.value().padded_shape()[-2] == TILE_HEIGHT,
                    "Beta tensor height must be TILE_HEIGHT (32), got: {}",
                    beta.value().padded_shape()[-2]);
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
}

spec_return_value_t LayerNormPostAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            args.output_dtype.value_or(input_tensor.dtype()),
            tt::tt_metal::PageConfig(Layout::TILE),
            args.memory_config));
}

tensor_return_value_t LayerNormPostAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::normalization::layernorm_post_all_gather

namespace ttnn::prim {
ttnn::operations::normalization::layernorm_post_all_gather::LayerNormPostAllGatherDeviceOperation::tensor_return_value_t
layernorm_post_all_gather(
    const Tensor& input,
    const Tensor& stats,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    ttnn::operations::normalization::layernorm::LayerNormDistributedType norm_type,
    float eps,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<bool>& use_2d_core_grid,
    const ttnn::operations::normalization::layernorm::LayerNormDistributedDefaultProgramConfig& program_config) {
    using OperationType =
        ttnn::operations::normalization::layernorm_post_all_gather::LayerNormPostAllGatherDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .norm_type = norm_type,
        .eps = eps,
        .memory_config = memory_config,
        .compute_kernel_config = compute_kernel_config,
        .output_dtype = output_dtype,
        .use_2d_core_grid = use_2d_core_grid,
        .program_config = program_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input = input,
        .stats = stats,
        .gamma = gamma,
        .beta = beta,
    };

    return ttnn::device_operation::detail::launch_on_device<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_post_all_gather_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::experimental::transformer::dit_layernorm {

PostAllGatherDeviceOperation::program_factory_t PostAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::PostAllGatherWelfordProgramFactory{};
}

void PostAllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void PostAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& stats = tensor_args.stats;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    TT_FATAL(!a.is_sharded(), "DIT layernorm post-all-gather does not support sharded inputs.");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16 || a.dtype() == DataType::BFLOAT8_B || a.dtype() == DataType::FLOAT32,
        "Input tensor must be BFLOAT16, BFLOAT8_B, or FLOAT32, got: {}",
        a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands must be on device.");
    TT_FATAL(a.buffer() != nullptr, "Operands must be allocated in buffers on device.");

    TT_FATAL(stats.layout() == Layout::TILE, "Stats tensor must have TILE layout, got: {}", stats.layout());
    TT_FATAL(
        stats.dtype() == DataType::BFLOAT16 || stats.dtype() == DataType::BFLOAT8_B ||
            stats.dtype() == DataType::FLOAT32,
        "Stats tensor must be BF16, BF8_B, or FLOAT32.");
    TT_FATAL(stats.storage_type() == StorageType::DEVICE, "Operands must be on device.");
    TT_FATAL(stats.buffer() != nullptr, "Operands must be allocated in buffers on device.");

    TT_FATAL(
        stats.padded_shape()[-1] % (2 * TILE_WIDTH) == 0,
        "Stats inner dimension must be multiple of 64 (two tiles), got: {}",
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
        TT_FATAL(beta.has_value(), "Beta must be provided when gamma is provided.");
        const auto& gamma_tensor = gamma.value();
        TT_FATAL(gamma_tensor.storage_type() == StorageType::DEVICE, "Gamma must be on device.");
        TT_FATAL(gamma_tensor.buffer() != nullptr, "Gamma must be allocated on device.");
        TT_FATAL(
            gamma_tensor.dtype() == DataType::BFLOAT16 || gamma_tensor.dtype() == DataType::FLOAT32,
            "Gamma must be BF16 or FLOAT32.");
        TT_FATAL(a.device() == gamma_tensor.device(), "Input and gamma tensors must be on same device");

        const auto& beta_tensor = beta.value();
        TT_FATAL(beta_tensor.storage_type() == StorageType::DEVICE, "Beta must be on device.");
        TT_FATAL(beta_tensor.buffer() != nullptr, "Beta must be allocated on device.");
        TT_FATAL(
            beta_tensor.dtype() == DataType::BFLOAT16 || beta_tensor.dtype() == DataType::FLOAT32,
            "Beta must be BF16 or FLOAT32.");
        TT_FATAL(a.device() == beta_tensor.device(), "Input and beta tensors must be on same device");

        auto check_layout = [&](const Tensor& t, const std::string& name) {
            if (t.layout() == Layout::TILE) {
                TT_FATAL(t.padded_shape()[-2] == TILE_HEIGHT, "{} height must be TILE_HEIGHT (32)", name);
                TT_FATAL(t.padded_shape()[-1] == a.padded_shape()[-1], "{} hidden dimension must match input.", name);
            } else {
                TT_FATAL(t.layout() == Layout::ROW_MAJOR, "{} must be TILE or ROW_MAJOR", name);
                TT_FATAL(
                    t.padded_shape()[-1] == TILE_WIDTH &&
                        t.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH,
                    "{} dimensions must align with input.",
                    name);
            }
        };
        check_layout(gamma_tensor, "Gamma");
        check_layout(beta_tensor, "Beta");
    } else {
        TT_FATAL(!beta.has_value(), "Beta must not be provided without gamma.");
    }
}

PostAllGatherDeviceOperation::spec_return_value_t PostAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(args.dtype.value_or(input_tensor.dtype()), PageConfig(Layout::TILE), args.memory_config));
}

PostAllGatherDeviceOperation::tensor_return_value_t PostAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::experimental::transformer::dit_layernorm

namespace ttnn::prim {

Tensor dit_layernorm_post_all_gather(
    const Tensor& input,
    const Tensor& stats,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype) {
    using OperationType = ttnn::operations::experimental::transformer::dit_layernorm::PostAllGatherDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .eps = eps,
            .memory_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .dtype = dtype,
        },
        OperationType::tensor_args_t{
            .input = input,
            .stats = stats,
            .gamma = gamma.has_value() ? std::make_optional(gamma.value()) : std::nullopt,
            .beta = beta.has_value() ? std::make_optional(beta.value()) : std::nullopt});
}

}  // namespace ttnn::prim

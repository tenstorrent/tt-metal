// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

namespace {
// True when the padded shape is the logical shape with only the last two dims rounded up to whole tiles.
bool batch_norm_is_minimally_tile_padded(const Tensor& t) {
    const auto& logical = t.logical_shape();
    const auto& padded = t.padded_shape();
    const auto& tile = t.tensor_spec().tile();
    if (logical.rank() != padded.rank()) {
        return false;
    }
    for (int d = 0; d < static_cast<int>(logical.rank()); ++d) {
        uint32_t expected = logical[d];
        if (d == static_cast<int>(logical.rank()) - 1) {
            expected = tt::round_up(expected, tile.get_width());
        } else if (d == static_cast<int>(logical.rank()) - 2) {
            expected = tt::round_up(expected, tile.get_height());
        }
        if (padded[d] != expected) {
            return false;
        }
    }
    return true;
}
}  // namespace

BatchNormDeviceOperation::program_factory_t BatchNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return BatchNormProgramFactory{};
}

void BatchNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to batch_norm need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to batch_norm need to be allocated in buffers on device!");
    TT_FATAL(input.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", input.layout());
    TT_FATAL(!input.is_sharded(), "Quasar batch_norm does not support sharded input.");
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(), "Quasar batch_norm does not support sharded output.");
    TT_FATAL(input.logical_shape().rank() == 4, "batch_norm supports tensors of rank 4");
    // The factory derives the per-plane tile count from the padded H x W, so an empty input has no geometry.
    TT_FATAL(
        input.logical_volume() > 0,
        "Quasar batch_norm does not support empty tensors (logical shape {})",
        input.logical_shape());
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16,
        "Quasar batch_norm: input must be BFLOAT16 (no BFLOAT8_B on Quasar; FLOAT32 not implemented), got: {}",
        input.dtype());
    TT_FATAL(
        input.tensor_spec().tile() == Tile{},
        "Quasar batch_norm requires standard 32x32 tiles, got {}x{}",
        input.tensor_spec().tile().get_height(),
        input.tensor_spec().tile().get_width());
    // The factory schedules every physical tile of every (n, c) plane, so the padded shape must be the logical
    // shape rounded up to whole tiles and nothing more.
    TT_FATAL(
        batch_norm_is_minimally_tile_padded(input),
        "Quasar batch_norm does not support padding beyond the last logical tile (logical {}, padded {})",
        input.logical_shape(),
        input.padded_shape());

    const uint32_t channels = input.logical_shape()[1];
    auto check_param = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "{} must be allocated in a buffer on device", name);
        TT_FATAL(t.device() == input.device(), "{} must be on the same device as the input", name);
        TT_FATAL(t.layout() == Layout::TILE, "{} must have TILE layout, got: {}", name, t.layout());
        TT_FATAL(t.dtype() == DataType::BFLOAT16, "Quasar batch_norm: {} must be BFLOAT16, got: {}", name, t.dtype());
        TT_FATAL(!t.is_sharded(), "{} must be interleaved", name);
        TT_FATAL(t.logical_shape().rank() == 4, "{} must have rank 4", name);
        TT_FATAL(
            t.logical_shape()[0] == 1 && t.logical_shape()[2] == 1 && t.logical_shape()[3] == 1,
            "{} must be [1, C, 1, 1], got {}",
            name,
            t.logical_shape());
        TT_FATAL(
            t.logical_shape()[1] == channels,
            "{}[1] ({}) must match the input's channel count ({})",
            name,
            t.logical_shape()[1],
            channels);
        // The reader fetches page c for channel c: exactly one standard tile per channel.
        TT_FATAL(
            t.tensor_spec().tile() == Tile{},
            "Quasar batch_norm requires standard 32x32 {} tiles, got {}x{}",
            name,
            t.tensor_spec().tile().get_height(),
            t.tensor_spec().tile().get_width());
        TT_FATAL(
            t.padded_shape() == Shape({1, channels, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
            "{} must have exactly one tile per channel (padded shape [1, C, 32, 32]), got padded shape {}",
            name,
            t.padded_shape());
    };
    check_param(tensor_args.running_mean, "running_mean");
    check_param(tensor_args.running_var, "running_var");
    if (tensor_args.weight.has_value()) {
        check_param(*tensor_args.weight, "weight");
    }
    if (tensor_args.bias.has_value()) {
        check_param(*tensor_args.bias, "bias");
    }
}

TensorSpec BatchNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Keep the input's page config and alignment: the factory schedules the input's physical tiles.
    const auto& input = tensor_args.input;
    const auto& in_layout = input.tensor_spec().tensor_layout();
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(
            input.dtype(),
            in_layout.get_page_config(),
            operation_attributes.output_mem_config,
            in_layout.get_alignment()));
}

Tensor BatchNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    float eps,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const ThreadPlan& threads) {
    return ttnn::device_operation::launch<BatchNormDeviceOperation>(
        BatchNormParams{
            .eps = eps,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .threads = threads,
        },
        BatchNormInputs{
            .input = input,
            .running_mean = running_mean,
            .running_var = running_var,
            .weight = weight,
            .bias = bias,
        });
}

}  // namespace ttnn::prim::qsr

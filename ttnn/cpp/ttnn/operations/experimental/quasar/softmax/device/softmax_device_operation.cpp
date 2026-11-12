// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

namespace {
// True when the padded shape is the logical shape with only the last two dims rounded up to whole tiles.
bool softmax_is_minimally_tile_padded(const Tensor& t) {
    const auto& logical = t.logical_shape();
    const auto& padded = t.padded_shape();
    const auto& tile = t.tensor_spec().tile();
    // A rank-1 TILE tensor carries an implicit tile-height dim: logical [W] pads to [tile_h, round_up(W)].
    if (logical.rank() == 1) {
        return padded.rank() == 2 && padded[0] == tile.get_height() &&
               padded[1] == tt::round_up(logical[0], tile.get_width());
    }
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

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return SoftmaxProgramFactory{};
}

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    // The factory derives the tile-row count from the padded width, so an empty input has no valid geometry.
    TT_FATAL(
        a.logical_volume() > 0, "Quasar softmax does not support empty tensors (logical shape {})", a.logical_shape());
    TT_FATAL(!a.is_sharded(), "Quasar softmax does not support sharded input.");
    TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Quasar softmax does not support sharded output.");
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16,
        "Quasar softmax: input must be BFLOAT16 (no BFLOAT8_B on Quasar; FLOAT32 not implemented), got: {}",
        a.dtype());
    TT_FATAL(
        a.tensor_spec().tile() == Tile{},
        "Quasar softmax requires standard 32x32 tiles, got {}x{}",
        a.tensor_spec().tile().get_height(),
        a.tensor_spec().tile().get_width());
    // Both reductions run over every physical tile of the row (no padding mask yet), so the softmax dimension
    // must be tile-aligned with no padded columns; the factory schedules the input's physical rows, so the
    // other dims must not be padded beyond the last logical tile either.
    TT_FATAL(
        a.logical_shape()[-1] % tt::constants::TILE_WIDTH == 0 && a.padded_shape()[-1] == a.logical_shape()[-1],
        "Quasar softmax: the last dimension ({}) must be a multiple of {} with no padded columns (padded {})",
        a.logical_shape()[-1],
        tt::constants::TILE_WIDTH,
        a.padded_shape()[-1]);
    TT_FATAL(
        softmax_is_minimally_tile_padded(a),
        "Quasar softmax does not support padding beyond the last logical tile (logical {}, padded {})",
        a.logical_shape(),
        a.padded_shape());
}

TensorSpec SoftmaxDeviceOperation::compute_output_specs(
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

Tensor SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor softmax(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool numeric_stable,
    const ThreadPlan& threads) {
    return ttnn::device_operation::launch<SoftmaxDeviceOperation>(
        SoftmaxParams{
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .numeric_stable = numeric_stable,
            .threads = threads,
        },
        SoftmaxInputs{.input = input_tensor});
}

}  // namespace ttnn::prim::qsr

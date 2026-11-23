// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation.hpp"

#include <cstdlib>
#include <cstring>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>

namespace ttnn::prim::qsr {

ReshapeViewDeviceOperation::program_factory_t ReshapeViewDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
        return ReshapeViewRowMajorProgramFactory{};
    }
    return ReshapeViewTiledProgramFactory{};
}

void ReshapeViewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const Tensor& input = tensor_args.input;
    const auto& in_shape = input.logical_shape();
    const auto& out_shape = operation_attributes.logical_output_shape;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.is_allocated(), "Operands need to be allocated in buffers on device!");
    // The kernels move elements as opaque bytes, so any fixed-width element works; block-float tiles
    // (shared exponents per 16 elements) cannot be re-tiled bytewise, and Quasar has no BFLOAT8_B at all.
    const DataType dtype = input.dtype();
    TT_FATAL(
        dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::INT32 ||
            dtype == DataType::UINT32 || dtype == DataType::UINT16 || dtype == DataType::UINT8,
        "Quasar reshape moves fixed-width elements only (bfloat16, float32, int32, uint32, uint16, uint8), got {}",
        dtype);
    TT_FATAL(
        operation_attributes.num_threads >= 1 && operation_attributes.num_threads <= RESHAPE_MAX_DM_THREADS,
        "Quasar reshape runs on 1..{} DM harts per node, got {}",
        RESHAPE_MAX_DM_THREADS,
        operation_attributes.num_threads);
    TT_FATAL(
        in_shape.volume() == out_shape.volume() && in_shape.volume() != 0,
        "Quasar reshape needs equal, non-zero volumes, got {} -> {}",
        in_shape,
        out_shape);

    if (input.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            in_shape.rank() == 2 && out_shape.rank() == 2,
            "Quasar ROW_MAJOR reshape works on rank-2 [rows, last_dim] views, got {} -> {}",
            in_shape,
            out_shape);
        // The row-major kernel addresses page r as row r of the flat stream, which is only the buffer's
        // page order when the tensor is interleaved.
        TT_FATAL(
            !input.memory_config().is_sharded() && !operation_attributes.output_mem_config.is_sharded(),
            "Quasar ROW_MAJOR reshape needs interleaved input and output (the host entry converts sharded tensors)");
        return;
    }

    TT_FATAL(
        input.layout() == Layout::TILE, "Quasar reshape supports ROW_MAJOR and TILE layouts, got {}", input.layout());
    TT_FATAL(
        in_shape.rank() == 3 && out_shape.rank() == 3,
        "Quasar TILE reshape works on rank-3 [B, H, W] views, got {} -> {}",
        in_shape,
        out_shape);
    TT_FATAL(
        input.tensor_spec().tile() == tt::tt_metal::Tile{},
        "Quasar TILE reshape requires standard 32x32 tiles with 16x16 faces, got {}x{}",
        input.tensor_spec().tile().get_height(),
        input.tensor_spec().tile().get_width());
    // The kernel computes tile ids from the logical shape rounded up to whole tiles.
    auto minimally_padded = [](const ttnn::Shape& logical, const ttnn::Shape& padded) {
        return padded.rank() == 3 && padded[0] == logical[0] && padded[1] == tt::round_up(logical[1], TILE_HEIGHT) &&
               padded[2] == tt::round_up(logical[2], TILE_WIDTH);
    };
    TT_FATAL(
        minimally_padded(in_shape, input.padded_shape()),
        "Quasar TILE reshape: input padded shape {} is not logical shape {} rounded up to whole tiles",
        input.padded_shape(),
        in_shape);
    TT_FATAL(
        minimally_padded(out_shape, operation_attributes.padded_output_shape),
        "Quasar TILE reshape: output padded shape {} is not logical shape {} rounded up to whole tiles",
        operation_attributes.padded_output_shape,
        out_shape);
}

ReshapeViewDeviceOperation::spec_return_value_t ReshapeViewDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        operation_attributes.logical_output_shape,
        tt::tt_metal::experimental::tensor_layout_from_padded_shape(
            tensor_args.input.dtype(),
            tt::tt_metal::PageConfig(tensor_args.input.layout()),
            operation_attributes.output_mem_config,
            operation_attributes.logical_output_shape,
            operation_attributes.padded_output_shape));
}

ttnn::Tensor ReshapeViewDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t ReshapeViewDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(operation_attributes, tensor_args);
    return tt::tt_metal::operation::hash_operation<ReshapeViewDeviceOperation>(
        operation_attributes.logical_output_shape,
        operation_attributes.padded_output_shape,
        operation_attributes.output_mem_config,
        operation_attributes.sub_core_grid.has_value(),
        operation_attributes.sub_core_grid.has_value() ? operation_attributes.sub_core_grid.value()
                                                       : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        operation_attributes.num_threads,
        operation_attributes.pad_word,  // a compile-time arg of the TILE kernel
        tensor_args,
        program_factory.index());
}

uint32_t reshape_threads_from_env() {
    uint32_t num_threads = RESHAPE_MAX_DM_THREADS;
    if (const char* env = std::getenv("TTNN_QSR_RESHAPE_THREADS")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        TT_FATAL(
            end != env && *end == '\0' && std::strchr(env, '-') == nullptr && parsed >= 1 &&
                parsed <= RESHAPE_MAX_DM_THREADS,
            "TTNN_QSR_RESHAPE_THREADS must be a hart count in 1..{}, got '{}'",
            RESHAPE_MAX_DM_THREADS,
            env);
        num_threads = static_cast<uint32_t>(parsed);
    }
    return num_threads;
}

ttnn::Tensor reshape_view(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const ttnn::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grid,
    uint32_t num_threads,
    uint32_t pad_word) {
    return ttnn::device_operation::launch<ReshapeViewDeviceOperation>(
        ReshapeViewParams{
            logical_output_shape, padded_output_shape, output_mem_config, sub_core_grid, num_threads, pad_word},
        ReshapeViewInputs{input});
}

}  // namespace ttnn::prim::qsr

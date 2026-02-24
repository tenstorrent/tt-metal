// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

TilizeWithValPaddingDeviceOperation::program_factory_t TilizeWithValPaddingDeviceOperation::select_program_factory(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor) {
    if (input_tensor.memory_config().is_sharded()) {
        bool use_optimized_sharding_program =
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
        use_optimized_sharding_program &=
            operation_attributes.output_mem_config.memory_layout() == input_tensor.memory_config().memory_layout();
        const auto& padded_shape = input_tensor.padded_shape();

        for (uint32_t i = 0; i < padded_shape.rank(); i++) {
            if (i != padded_shape.rank() - 2) {
                use_optimized_sharding_program &= padded_shape[i] == operation_attributes.output_padded_shape[i];
            }
        }
        if (use_optimized_sharding_program) {
            TT_FATAL(
                !operation_attributes.sub_core_grids.has_value(),
                "Sharded tilize does not support sub core grid specification");
            return TilizeWithValPaddingMultiCoreShardedFactory{};
        }
        return TilizeWithValPaddingMultiCoreDefaultFactory{};
    }
    if (!operation_attributes.enough_space_height) {
        return TilizeWithValPaddingMultiCoreBlockInterleavedFactory{};
    }
    if (!operation_attributes.use_multicore) {
        return TilizeWithValPaddingSingleCoreFactory{};
    }
    auto* device = input_tensor.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid =
        operation_attributes.sub_core_grids.has_value() ? operation_attributes.sub_core_grids.value() : default_grid;
    uint32_t num_blocks = operation_attributes.output_padded_shape.volume() /
                          operation_attributes.output_padded_shape[-1] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_per_row = operation_attributes.output_padded_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t num_tiles_per_col = operation_attributes.output_padded_shape[-2] / tt::constants::TILE_HEIGHT;

    size_t grid_area = available_grid.num_cores();
    auto [ncores, nblocks_per_core] = compute_ncores(grid_area, num_blocks);
    constexpr uint32_t threshold_row_block = 32;
    if (num_tiles_per_row > threshold_row_block &&
        (num_tiles_per_col > threshold_row_block || num_tiles_per_row > num_tiles_per_col)) {
        uint32_t num_blocks_block = (input_tensor.padded_shape()[-1] * input_tensor.padded_shape()[-2]) /
                                    (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
        auto ncores_wh = compute_ncores_wh(grid_area, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
        if (ncores < ncores_wh.ncores) {
            return TilizeWithValPaddingMultiCoreBlockInterleavedFactory{};
        }
    }
    return TilizeWithValPaddingMultiCoreDefaultFactory{};
}

void TilizeWithValPaddingDeviceOperation::validate_on_program_cache_miss(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor) {
    const auto& input_shape = input_tensor.logical_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16 or input_tensor.dtype() == DataType::INT32 or
            input_tensor.dtype() == DataType::UINT32 or input_tensor.dtype() == DataType::FLOAT32 or
            input_tensor.dtype() == DataType::UINT16,
        "Can only tilize bfloat16/float32 or int32/uint32/uint16 tensors");

    TT_FATAL(input_shape.rank() >= 1, "Input tensor must be of rank >= 1, but its shape is {}", input_shape);

    if (input_shape.rank() == 1) {
        // Special case: if input tensor is 1D row-major, output tiled tensor will have 1D logical shape
        // but 2D padded shape
        TT_FATAL(
            input_shape[0] <= operation_attributes.output_padded_shape[-1],
            "Output tensor shape {} must be greater than or equal to input shape {} in each dimension, but is "
            "smaller in dimension {}",
            operation_attributes.output_padded_shape,
            input_shape,
            0);
    } else {
        for (auto i = 0; i < input_shape.rank(); i++) {
            TT_FATAL(
                input_shape[i] <= operation_attributes.output_padded_shape[i],
                "Output tensor shape {} must be greater than or equal to input shape {} in each dimension, but is "
                "smaller in dimension {}",
                operation_attributes.output_padded_shape,
                input_shape,
                i);
        }
    }

    uint32_t num_rows = operation_attributes.output_padded_shape[-1];
    uint32_t inner_dim = operation_attributes.output_padded_shape[-2];
    TT_FATAL(
        inner_dim % TILE_WIDTH == 0 && num_rows % TILE_HEIGHT == 0,
        "To be tilizable output tensor shape {} must be divisible by tile size ({}, {})",
        operation_attributes.output_padded_shape,
        TILE_WIDTH,
        TILE_HEIGHT);

    if (input_tensor.memory_config().is_sharded()) {
        uint32_t element_size_in_bytes = input_tensor.element_size();
        uint32_t shard_width = input_tensor.shard_spec().has_value()
                                   ? input_tensor.shard_spec().value().shape[1]
                                   : input_tensor.nd_shard_spec().value().shard_shape[-1];

        const uint32_t page_size_bytes = shard_width * element_size_in_bytes;
        const uint32_t alignment_requirement = hal::get_l1_alignment();
        TT_FATAL(
            page_size_bytes % alignment_requirement == 0,
            "Input row-major shard width {} gives page size {} bytes, which must be aligned to {} bytes L1 SRAM "
            "buffer alignment "
            "requirement",
            shard_width,
            page_size_bytes,
            alignment_requirement);  // The shard_width must be an aligned size, or we will face alignment issues when
                                     // the reader tries to write to the CB, since we might write to unaligned
                                     // addresses.
    }
}

TensorSpec TilizeWithValPaddingDeviceOperation::compute_output_specs(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor) {
    const auto& input_shape = input_tensor.logical_shape();

    if (input_tensor.memory_config().is_sharded() && input_tensor.shard_spec().has_value() &&
        operation_attributes.output_mem_config.shard_spec().has_value()) {
        // This case only applies when we expect that the input and output are both legacy 2D sharded.
        // This bit forces the output tensor to be width-sharded.
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] =
            operation_attributes.output_padded_shape.volume() / operation_attributes.output_padded_shape[-1];
        auto mem_config = operation_attributes.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(
            input_shape,
            TensorLayout::fromPaddedShape(
                operation_attributes.output_dtype,
                PageConfig(Layout::TILE),
                mem_config,
                input_shape,
                operation_attributes.output_padded_shape));
    }

    return TensorSpec(
        input_shape,
        TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE),
            operation_attributes.output_mem_config,
            input_shape,
            operation_attributes.output_padded_shape));
}

Tensor TilizeWithValPaddingDeviceOperation::create_output_tensors(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor) {
    return create_device_tensor(compute_output_specs(operation_attributes, input_tensor), input_tensor.device());
}

Tensor tilize_with_val_padding(
    const Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const PadValue& pad_value,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DataType>& output_dtype,
    bool use_multicore,
    bool enough_space_width,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<TilizeWithValPaddingDeviceOperation>(
        TilizeWithValPaddingParams{
            .output_padded_shape = output_padded_shape,
            .pad_value = pad_value,
            .output_mem_config = output_mem_config.value_or(input_tensor.memory_config()),
            .output_dtype = output_dtype.value_or(input_tensor.dtype()),
            .use_multicore = use_multicore,
            .enough_space_width = enough_space_width,
            .enough_space_height = enough_space_height,
            .sub_core_grids = sub_core_grids,
        },
        input_tensor);
}
}  // namespace ttnn::prim

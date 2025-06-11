// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_layout_op.hpp"

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <tt-metalium/constants.hpp>
#include "cpp/ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

namespace detail {

bool requires_padding_change(const ttnn::Tensor& tensor, ttnn::Layout layout) {
    auto tile = tensor.tensor_spec().tile();
    if (layout == Layout::ROW_MAJOR) {
        // There shouldn't be extra paddings for Row Major layout
        return tensor.logical_shape() != tensor.padded_shape();
    }
    // It's okay for conversion to tile layout to preserve arbitrary padding as long as it satisfies the alignment
    TensorSpec padded_spec(
        tensor.padded_shape(),
        tt::tt_metal::TensorLayout(
            tensor.dtype(), tt::tt_metal::PageConfig(layout, std::move(tile)), tensor.memory_config()));
    return tensor.padded_shape() != padded_spec.padded_shape();
}

template <typename T>
Tensor to_layout_impl(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    T* device) {
    if (tensor_arg.layout() == layout) {
        if (dtype.has_value() and dtype.value() != tensor_arg.dtype()) {
            log_warning(
                tt::LogOp,
                "ttnn::to_layout: dtype is specified but the tensor is already in the requested layout! So, the "
                "dtype "
                "won't be changed!");
        }
        if (memory_config.has_value() and memory_config.value() != get_memory_config(tensor_arg).value()) {
            log_warning(
                tt::LogOp,
                "ttnn::to_layout: memory_config is specified but the tensor is already in the requested layout! "
                "So, "
                "the memory_config won't be changed!");
        }
        return tensor_arg;
    }

    const std::set<ttnn::Layout> supported_layouts = {
        ttnn::ROW_MAJOR_LAYOUT,
        ttnn::TILE_LAYOUT,
    };

    if (supported_layouts.find(layout) == supported_layouts.end()) {
        TT_THROW("ttnn::to_layout: Unsupported layout conversion from {} to {}!", tensor_arg.layout(), layout);
    }

    auto tensor = tensor_arg;
    const auto tile = tensor.tensor_spec().tile();
    auto output_shape = tensor_arg.logical_shape();
    auto output_memory_config =
        memory_config.value_or(ttnn::get_memory_config(tensor).value_or(ttnn::DRAM_MEMORY_CONFIG));

    TensorSpec tile_spec(
        tensor_arg.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_arg.dtype(), tt::tt_metal::PageConfig(Layout::TILE, tile), output_memory_config));
    auto padded_output_shape = tile_spec.padded_shape();
    auto original_rank = tensor_arg.logical_shape().rank();
    auto original_shape = tensor_arg.logical_shape();

    if (layout == ttnn::TILE_LAYOUT) {
        if (tensor.padded_shape().size() < 2) {
            SmallVector<uint32_t> new_padded_shape(2, 1);
            new_padded_shape[1] = tensor.padded_shape()[-1];
            new_padded_shape[0] = tensor.padded_shape()[-2];
            tensor = ttnn::experimental::view(tensor, tensor.logical_shape(), Shape(new_padded_shape));
        }
    }

    if (tt::tt_metal::is_device_tensor(tensor_arg)) {
        bool use_multicore_untilize = true;
        bool use_multicore_tilize = true;

        if (not requires_padding_change(tensor, layout)) {
            if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
                return ttnn::untilize(tensor, output_memory_config, use_multicore_untilize);
            } else if (layout == ttnn::TILE_LAYOUT) {
                if (tensor.is_sharded()) {
                    const auto tensor_tile = tensor.tensor_spec().tile();
                    uint32_t tile_height = tensor_tile.get_height();
                    uint32_t tile_width = tensor_tile.get_width();
                    const auto shard_shape = get_memory_config(tensor).value().shard_spec().value().shape;
                    if (shard_shape[0] % tile_height != 0 or shard_shape[1] % tile_width != 0) {
                        TT_THROW(
                            "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of "
                            "TILE_SIZE!");
                    }
                }
                return ttnn::tilize(tensor, output_memory_config, dtype, use_multicore_tilize);
            } else {
                throw std::runtime_error("ttnn::to_layout: Unsupported layout!");
            }
        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            TT_FATAL(
                !dtype.has_value() || dtype.value() == tensor_arg.dtype(),
                "dtype cannot be different from tensor dtype when converting to ROW_MAJOR_LAYOUT on device!");

            if (tensor.is_sharded()) {
                const auto memory_config = tensor.memory_config();
                output_memory_config =
                    tt::tt_metal::MemoryConfig{memory_config.memory_layout(), memory_config.buffer_type()};
            }
            Shape output_tensor_end(SmallVector<uint32_t>(tensor.logical_shape().rank(), 0));
            int logical_rank = tensor.logical_shape().rank();
            for (int index = -1; index >= -logical_rank; --index) {
                output_tensor_end[index] = tensor.logical_shape()[index] - 1;
            }

            tensor =
                ttnn::untilize_with_unpadding(tensor, output_tensor_end, output_memory_config, use_multicore_untilize);
            return ttnn::reshape(tensor, ttnn::Shape{output_shape});

        } else if (layout == ttnn::TILE_LAYOUT) {
            if (tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                // ttnn::tilize_with_val_padding doesn't support height sharded tensors
                // workaround by applying padding and then tilizing
                SmallVector<std::pair<uint32_t, uint32_t>> padding = {
                    {0, 0},
                    {0, 0},
                    {0, padded_output_shape[2] - output_shape[2]},
                    {0, padded_output_shape[3] - output_shape[3]}};
                tensor = ttnn::pad(tensor, padding, 0, true, std::nullopt);
                return ttnn::tilize(tensor, output_memory_config, dtype, use_multicore_tilize);
            } else {
                PadValue pad_value_variant;
                if (tensor.dtype() == ttnn::DataType::BFLOAT16 or tensor.dtype() == ttnn::DataType::FLOAT32) {
                    pad_value_variant = 0.0f;
                } else {
                    pad_value_variant = (uint32_t)0;
                }

                tensor = ttnn::tilize_with_val_padding(
                    tensor,
                    Shape(padded_output_shape),
                    pad_value_variant,
                    output_memory_config,
                    dtype,
                    use_multicore_tilize);
            }
            if (original_rank == 1) {
                return ttnn::reshape(tensor, original_shape);
            }

            return ttnn::reshape(tensor, output_shape, padded_output_shape);
        } else {
            TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
        }
    } else {
        TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting layout on host!");
        if (not requires_padding_change(tensor, layout)) {
            return device ? tensor.to_layout(layout, device) : tensor.to_layout(layout);
        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            tensor = device ? tensor.to_layout(layout, device) : tensor.to_layout(layout);
            tensor = tensor.unpad_from_tile(tensor.logical_shape());
            return ttnn::reshape(tensor, ttnn::Shape{output_shape});
        } else if (layout == ttnn::TILE_LAYOUT) {
            SmallVector<uint32_t> padded_input_start;
            for (int index = 0; index < padded_output_shape.rank(); ++index) {
                padded_input_start.push_back(0);
            }
            tensor = tensor.pad(ttnn::Shape(padded_output_shape), ttnn::Shape(std::move(padded_input_start)), 0);
            tensor = device ? tensor.to_layout(layout, device) : tensor.to_layout(layout);
            return ttnn::experimental::view(tensor, output_shape, padded_output_shape);
        } else {
            TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
        }
    }
}
}  // namespace detail

/* static */ Tensor ToLayout::invoke(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    IDevice* device) {
    return detail::to_layout_impl(tensor_arg, layout, dtype, memory_config, device);
}

/* static */ Tensor ToLayout::invoke(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    MeshDevice* device) {
    return detail::to_layout_impl(tensor_arg, layout, dtype, memory_config, device);
}

}  // namespace core

}  // namespace operations

}  // namespace ttnn

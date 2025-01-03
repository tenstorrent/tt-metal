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
#include "tt_metal/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

namespace detail {

inline bool validate_nd_support(const ttnn::Tensor& tensor_arg, const ttnn::Layout layout) {
    const auto initial_shape = tensor_arg.get_shape();
    if (initial_shape.rank() > 4 && tensor_arg.get_layout() != layout) {
        for (int i = 0; i < initial_shape.rank() - 4; i++) {
            TT_FATAL(
                initial_shape[i] == 1,
                "For ND tensors, shape dimensions greater than 4 should be 1, shape at index{} is {}",
                i,
                initial_shape[i]);
        }
    }
    return true;
}

// Issue #8617: Limitations on tensor width for multicore device tilize
inline bool use_multicore_device_tilize(
    const Tensor& input, const std::optional<tt::tt_metal::DataType>& output_dtype) {
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);

    uint32_t output_single_tile_size =
        output_dtype.has_value()
            ? tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
            : input_single_tile_size;

    uint32_t num_tiles_in_row = input.get_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t max_l1_size =
        input.device()->l1_size_per_core() / 2 - input.device()->get_base_allocator_addr(HalMemType::L1);
    uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs

    return num_tiles_in_row <= max_tiles;
}

template <typename T>
Tensor to_layout_impl(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    T* device) {
    if (tensor_arg.get_layout() == layout) {
        if (dtype.has_value() and dtype.value() != tensor_arg.get_dtype()) {
            tt::log_warning(
                tt::LogOp,
                "ttnn::to_layout: dtype is specified but the tensor is already in the requested layout! So, the "
                "dtype "
                "won't be changed!");
        }
        if (memory_config.has_value() and memory_config.value() != get_memory_config(tensor_arg).value()) {
            tt::log_warning(
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
        TT_THROW("ttnn::to_layout: Unsupported layout conversion from {} to {}!", tensor_arg.get_layout(), layout);
    }

    const auto requires_padding_change =
        [](ttnn::Tensor& tensor, ttnn::Layout layout, const ttnn::Shape& shape) -> bool {
        const auto intended_shape = shape;
        const auto padded_shape = shape.with_tile_padding();
        if (layout == ttnn::ROW_MAJOR_LAYOUT and intended_shape != padded_shape) {
            return true;
        }
        if (layout == ttnn::TILE_LAYOUT) {
            auto tile_shape = tensor.tensor_spec().tile().get_tile_shape();
            if (padded_shape.rank() < 2 or padded_shape[-1] % tile_shape[1] != 0 or
                padded_shape[-2] % tile_shape[0] != 0) {
                return true;
            }
        }
        return false;
    };

    const auto intended_shape = tensor_arg.get_shape();

    auto tensor = tensor_arg;
    const auto tile = tensor.get_tensor_spec().tile();

    SmallVector<uint32_t> output_shape;
    if (layout == ttnn::TILE_LAYOUT and intended_shape.rank() < 2) {
        output_shape.push_back(1);
        tensor = ttnn::reshape(
            tensor,
            ttnn::Shape(
                SmallVector<uint32_t>{1, intended_shape[0]},
                SmallVector<uint32_t>{1, tensor_arg.get_shape().with_tile_padding()[0]}));
    }
    for (auto index = 0; index < intended_shape.rank(); ++index) {
        output_shape.push_back(intended_shape[index]);
    }

    auto padded_output_shape = output_shape;
    for (auto index = output_shape.size() - 2; index < output_shape.size(); ++index) {
        padded_output_shape[index] = ttnn::pad_to_multiple_of_tile_size(
            padded_output_shape[index],
            (index == output_shape.size() - 2) ? tile.get_tile_shape()[0] : tile.get_tile_shape()[1]);
    }

    auto output_memory_config =
        memory_config.value_or(ttnn::get_memory_config(tensor).value_or(ttnn::DRAM_MEMORY_CONFIG));

    if (ttnn::is_tensor_on_device_or_multidevice(tensor_arg)) {
        bool use_multicore_untilize = true;
        bool use_multicore_tilize = use_multicore_device_tilize(tensor, dtype);

        if (not requires_padding_change(tensor, layout, tensor.get_shape())) {
            if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
                validate_nd_support(tensor_arg, layout);
                return ttnn::untilize(tensor, output_memory_config, use_multicore_untilize);
            } else if (layout == ttnn::TILE_LAYOUT) {
                if (tensor.is_sharded()) {
                    const auto shard_shape = get_memory_config(tensor).value().shard_spec.value().shape;
                    if (shard_shape[0] % ttnn::TILE_SIZE != 0 or shard_shape[1] % ttnn::TILE_SIZE != 0) {
                        TT_THROW(
                            "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of "
                            "TILE_SIZE!");
                    }
                }
                validate_nd_support(tensor_arg, layout);
                return ttnn::tilize(tensor, output_memory_config, dtype, use_multicore_tilize);
            } else {
                throw std::runtime_error("ttnn::to_layout: Unsupported layout!");
            }
        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");

            if (tensor.is_sharded()) {
                const auto memory_config = tensor.memory_config();
                output_memory_config =
                    tt::tt_metal::MemoryConfig{memory_config.memory_layout, memory_config.buffer_type};
            }
            SmallVector<uint32_t> output_tensor_end;
            for (auto index = 0; index < tensor.get_shape().rank(); ++index) {
                output_tensor_end.push_back(tensor.get_shape()[index] - 1);
            }

            validate_nd_support(tensor_arg, layout);
            tensor =
                ttnn::untilize_with_unpadding(tensor, output_tensor_end, output_memory_config, use_multicore_untilize);
            return ttnn::reshape(tensor, ttnn::SimpleShape{output_shape});

        } else if (layout == ttnn::TILE_LAYOUT) {
            SmallVector<uint32_t> padded_output_shape;

            for (int index = 0; index < tensor.get_shape().rank(); ++index) {
                uint32_t second_last_rank = tensor.get_shape().rank() - 2;  // h dim
                uint32_t padded_value =
                    index < second_last_rank
                        ? tensor.get_shape()[index]
                        : ttnn::pad_to_multiple_of_tile_size(
                              tensor.get_shape()[index],
                              index == second_last_rank ? tile.get_tile_shape()[0] : tile.get_tile_shape()[1]);
                padded_output_shape.push_back(padded_value);
            }
            if (tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                // ttnn::tilize_with_val_padding doesn't support height sharded tensors
                // workaround by applying padding and then tilizing
                SmallVector<std::pair<uint32_t, uint32_t>> padding = {
                    {0, 0},
                    {0, 0},
                    {0, padded_output_shape[2] - output_shape[2]},
                    {0, padded_output_shape[3] - output_shape[3]}};
                tensor = ttnn::pad(0, tensor, padding, 0, true, std::nullopt);
                validate_nd_support(tensor_arg, layout);
                return ttnn::tilize(tensor, output_memory_config, dtype, use_multicore_tilize);
            } else {
                PadValue pad_value_variant;
                if (tensor.get_dtype() == ttnn::DataType::BFLOAT16 or tensor.get_dtype() == ttnn::DataType::FLOAT32) {
                    pad_value_variant = 0.0f;
                } else {
                    pad_value_variant = (uint32_t)0;
                }

                validate_nd_support(tensor_arg, layout);
                tensor = ttnn::tilize_with_val_padding(
                    tensor, padded_output_shape, pad_value_variant, output_memory_config, dtype, use_multicore_tilize);
            }

            return ttnn::reshape(tensor, ttnn::Shape(tt::tt_metal::LegacyShape{output_shape, padded_output_shape}));

        } else {
            TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
        }
    } else {
        TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting layout on host!");
        if (not requires_padding_change(tensor, layout, tensor.get_shape())) {
            return device ? tensor.to(layout, device) : tensor.to(layout);
        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            tensor = device ? tensor.to(layout, device) : tensor.to(layout);
            tensor = tensor.unpad_from_tile(tensor.get_logical_shape());
            return ttnn::reshape(tensor, ttnn::SimpleShape{output_shape});
        } else if (layout == ttnn::TILE_LAYOUT) {
            SmallVector<uint32_t> padded_output_shape;
            SmallVector<uint32_t> padded_input_start;
            for (int index = 0; index < tensor.get_shape().rank(); ++index) {
                uint32_t second_last_rank = tensor.get_shape().rank() - 2;  // h dim
                uint32_t padded_value =
                    index < second_last_rank
                        ? tensor.get_shape()[index]
                        : ttnn::pad_to_multiple_of_tile_size(
                              tensor.get_shape()[index],
                              index == second_last_rank ? tile.get_tile_shape()[0] : tile.get_tile_shape()[1]);
                padded_output_shape.push_back(padded_value);
                padded_input_start.push_back(0);
            }
            tensor =
                tensor.pad(ttnn::SimpleShape(padded_output_shape), ttnn::SimpleShape(std::move(padded_input_start)), 0);
            tensor = device ? tensor.to(layout, device) : tensor.to(layout);
            return ttnn::reshape(tensor, ttnn::Shape(tt::tt_metal::LegacyShape{output_shape, padded_output_shape}));
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
    Device* device) {
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

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

bool requires_padding_change(const ttnn::Tensor& tensor, ttnn::Layout layout) {
    auto tile = tensor.get_tensor_spec().tile();
    if (layout == Layout::ROW_MAJOR) {
        // There shouldn't be extra paddings for Row Major layout
        return tensor.logical_shape() != tensor.padded_shape();
    }
    // It's okay for conversion to tile layout to preserve arbitrary padding as long as it satisfies the alignment
    TensorSpec padded_spec(
        tensor.padded_shape(),
        TensorLayout(tensor.dtype(), PageConfig(layout, std::move(tile)), tensor.memory_config()));
    return tensor.get_padded_shape() != padded_spec.padded_shape();
}

template <typename T>
Tensor to_layout_impl_on_device(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    ttnn::MemoryConfig output_memory_config,
    T* device) {
    bool use_multicore_untilize = true;
    bool use_multicore_tilize = use_multicore_device_tilize(tensor_arg, dtype);

    if (!requires_padding_change(tensor_arg, layout)) {
        if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            TT_FATAL(
                !dtype.has_value() || dtype.value() == tensor_arg.dtype(),
                "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
            return ttnn::untilize(tensor_arg, output_memory_config, use_multicore_untilize);
        }
        return ttnn::tilize(tensor_arg, output_memory_config, dtype, use_multicore_tilize);
    }

    auto tensor_shape = tensor_arg.get_logical_shape();

    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        TT_FATAL(
            !dtype.has_value() || dtype.value() == tensor_arg.dtype(),
            "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");

        if (tensor_arg.is_sharded()) {
            const auto memory_config = tensor_arg.memory_config();
            output_memory_config = tt::tt_metal::MemoryConfig{memory_config.memory_layout, memory_config.buffer_type};
        }
        SmallVector<uint32_t> output_tensor_end;
        for (auto index = 0; index < tensor_shape.rank(); ++index) {
            output_tensor_end.push_back(tensor_shape[index] - 1);
        }

        auto tensor =
            ttnn::untilize_with_unpadding(tensor_arg, output_tensor_end, output_memory_config, use_multicore_untilize);
        return ttnn::reshape(tensor, tensor_shape);
    }

    TensorSpec result_spec(
        tensor_arg.logical_shape(),
        TensorLayout(
            tensor_arg.dtype(),
            PageConfig(layout, std::move(tensor_arg.tensor_spec().tile())),
            tensor_arg.memory_config()));

    // ttnn::tilize_with_val_padding doesn't support height sharded tensors
    // workaround by applying padding and then tilizing
    if (tensor_arg.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        ttnn::SmallVector<std::pair<uint32_t, uint32_t>> pad(result_spec.shape().rank());
        auto output_padding = result_spec.shape().padding();
        for (size_t i = 0; i < result_spec.padded_shape().rank(); i++) {
            pad[i] = {output_padding[i].front, output_padding[i].back};
        }
        auto tensor = ttnn::pad(0, tensor_arg, tt::stl::Span(pad), 0, true, std::nullopt);
        return ttnn::tilize(tensor, output_memory_config, dtype, use_multicore_tilize);
    }

    PadValue pad_value_variant;
    if (tensor_arg.get_dtype() == ttnn::DataType::BFLOAT16 or tensor_arg.get_dtype() == ttnn::DataType::FLOAT32) {
        pad_value_variant = 0.0f;
    } else {
        pad_value_variant = (uint32_t)0;
    }

    auto tensor = ttnn::tilize_with_val_padding(
        tensor_arg, result_spec.padded_shape(), pad_value_variant, output_memory_config, dtype, use_multicore_tilize);
    return tensor.reshape(tensor_arg.logical_shape());
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

    if (layout != ROW_MAJOR_LAYOUT && layout != TILE_LAYOUT) {
        TT_THROW("ttnn::to_layout: Unsupported layout conversion from {} to {}!", tensor_arg.get_layout(), layout);
    }

    auto output_memory_config =
        memory_config.value_or(ttnn::get_memory_config(tensor_arg).value_or(ttnn::DRAM_MEMORY_CONFIG));

    if (ttnn::is_tensor_on_device_or_multidevice(tensor_arg)) {
        return to_layout_impl_on_device(tensor_arg, layout, dtype, std::move(output_memory_config), device);
    }

    TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting layout on host!");
    if (not requires_padding_change(tensor_arg, layout)) {
        return device ? tensor_arg.to(layout, device) : tensor_arg.to(layout);
    }

    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        auto tensor = device ? tensor_arg.to(layout, device) : tensor_arg.to(layout);
        tensor = tensor.unpad_from_tile(tensor.get_logical_shape());
        return tensor.reshape(tensor_arg.logical_shape());
    }

    SmallVector<uint32_t> padded_input_start;
    for (int index = 0; index < tensor_arg.get_logical_shape().rank(); ++index) {
        padded_input_start.push_back(0);
    }
    TensorSpec result_spec(
        tensor_arg.padded_shape(),
        TensorLayout::fromPaddedShape(
            tensor_arg.dtype(),
            PageConfig(layout, std::move(tensor_arg.tensor_spec().tile())),
            tensor_arg.memory_config(),
            tensor_arg.logical_shape(),
            tensor_arg.padded_shape()));

    auto tensor = tensor_arg.pad(result_spec.padded_shape(), ttnn::SimpleShape(std::move(padded_input_start)), 0);
    tensor = device ? tensor.to(layout, device) : tensor.to(layout);
    return tensor.reshape(result_spec.logical_shape());
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

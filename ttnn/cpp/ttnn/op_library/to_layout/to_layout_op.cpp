// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/op_library/to_layout/to_layout_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_eager/tt_dnn/op_library/untilize/untilize_op.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"
namespace ttnn {

namespace operations {

namespace core {

namespace detail {
template <typename T>
Tensor execute(
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

    const auto requires_padding_change = [](ttnn::Layout layout, const ttnn::Shape& shape) -> bool {
        const auto intended_shape = shape;
        const auto padded_shape = shape.with_tile_padding();
        if (layout == ttnn::ROW_MAJOR_LAYOUT and intended_shape != padded_shape) {
            return true;
        } else if (
            layout == ttnn::TILE_LAYOUT and (padded_shape.rank() < 2 or padded_shape[-1] % ttnn::TILE_SIZE != 0 or
                                             padded_shape[-2] % ttnn::TILE_SIZE != 0)) {
            return true;
        } else {
            return false;
        }
    };

    const auto intended_shape = tensor_arg.get_shape();

    std::vector<uint32_t> output_shape;
    if (layout == ttnn::TILE_LAYOUT and intended_shape.rank() < 2) {
        output_shape.push_back(1);
    }
    for (auto index = 0; index < intended_shape.rank(); ++index) {
        output_shape.push_back(intended_shape[index]);
    }

    auto padded_output_shape = output_shape;
    for (auto index = output_shape.size() - 2; index < output_shape.size(); ++index) {
        padded_output_shape[index] = ttnn::pad_to_multiple_of_tile_size(padded_output_shape[index]);
    }

    auto tensor = tensor_arg;

    auto output_memory_config =
        memory_config.value_or(ttnn::get_memory_config(tensor).value_or(ttnn::DRAM_MEMORY_CONFIG));

    if (ttnn::is_tensor_on_device_or_multidevice(tensor_arg)) {
        if (not requires_padding_change(layout, tensor.get_shape())) {
            bool use_multicore = true;
            if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
                return tt::tt_metal::untilize(tensor, output_memory_config, use_multicore);
            } else if (layout == ttnn::TILE_LAYOUT) {
                if (tensor.is_sharded()) {
                    const auto shard_shape = get_memory_config(tensor).value().shard_spec.value().shape;
                    if (shard_shape[0] % ttnn::TILE_SIZE != 0 or shard_shape[1] % ttnn::TILE_SIZE != 0) {
                        TT_THROW(
                            "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of "
                            "TILE_SIZE!");
                    }
                }
                return tt::tt_metal::tilize(tensor, output_memory_config, dtype, use_multicore);
            } else {
                throw runtime_error("ttnn::to_layout: Unsupported layout!");
            }
        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");

            if (tensor.is_sharded()) {
                const auto memory_layout_config = tensor.memory_config();
                output_memory_config =
                    tt::tt_metal::MemoryConfig{memory_layout_config.memory_layout, tt::tt_metal::BufferType::L1};
            }

            tensor = unsqueeze_to_4D(tensor);
            std::vector<uint32_t> output_tensor_end;
            for (auto index = 0; index < tensor.get_shape().rank(); ++index) {
                output_tensor_end.push_back(tensor.get_shape()[index] - 1);
            }

            tensor =
                tt::tt_metal::untilize_with_unpadding(tensor, {0, 0, 0, 0}, output_tensor_end, output_memory_config);
            return reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape}));

        } else if (layout == ttnn::TILE_LAYOUT) {
            tensor = unsqueeze_to_4D(tensor);
            std::vector<uint32_t> padded_4D_output_shape;
            padded_4D_output_shape.push_back(tensor.get_shape()[-4]);
            padded_4D_output_shape.push_back(tensor.get_shape()[-3]);
            padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-2]));
            padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-1]));
            tensor = tt::tt_metal::tilize_with_val_padding(
                tensor, padded_4D_output_shape, {0, 0, 0, 0}, 0, output_memory_config, dtype);
            return reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape, padded_output_shape}));

        } else {
            TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
        }
    } else {
        TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting layout on host!");
        if (not requires_padding_change(layout, tensor.get_shape())) {
            return device ? tensor.to(layout, device) : tensor.to(layout);
        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            tensor = unsqueeze_to_4D(tensor);
            tensor = device ? tensor.to(layout, device) : tensor.to(layout);
            tensor = tensor.unpad_from_tile(tensor.get_shape().value().without_padding());
            return reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape}));

        } else if (layout == ttnn::TILE_LAYOUT) {
            tensor = unsqueeze_to_4D(tensor);
            std::vector<uint32_t> padded_4D_output_shape;
            padded_4D_output_shape.push_back(tensor.get_shape()[-4]);
            padded_4D_output_shape.push_back(tensor.get_shape()[-3]);
            padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-2]));
            padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-1]));
            tensor = tensor.pad(padded_4D_output_shape, {0, 0, 0, 0}, 0);
            tensor = device ? tensor.to(layout, device) : tensor.to(layout);
            return reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape, padded_output_shape}));

        } else {
            TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
        }
    }
}
}  // namespace detail

/* static */ Tensor ToLayout::execute(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    Device* device) {
    return detail::execute(tensor_arg, layout, dtype, memory_config, device);
}

/* static */ Tensor ToLayout::execute(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    DeviceMesh* device) {
    return detail::execute(tensor_arg, layout, dtype, memory_config, device);
}

}  // namespace core

}  // namespace operations

}  // namespace ttnn

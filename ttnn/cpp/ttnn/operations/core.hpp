// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_eager/tt_dnn/op_library/move/move_op.hpp"
#include "tt_eager/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_eager/tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_eager/tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
// #include "ttnn/op_library/to_layout/to_layout_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {

namespace operations {
namespace core {

static inline const std::array<ttnn::TensorSchema, 1> reshape_input_schemas{
    ttnn::TensorSchema{
        1,
        8,
        {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16, ttnn::uint32, ttnn::int32, ttnn::float32},
        {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
        true,
        true,
        false},
};

inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    ttnn::validate_input_tensor("ttnn.reshape", tensor, reshape_input_schemas[0]);

    auto tensor_shape = tensor.get_shape();
    if (tensor_shape == shape) {
        return tensor;
    }

    const auto layout = tensor.get_layout();

    if (layout == ttnn::Layout::ROW_MAJOR) {
        if (tensor.is_contiguous()) {
            if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
                // Page size depends on the width, so only modify the shape if the width is the same
                if (tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]) {
                    return tensor.reshape(shape.value());
                }
            } else {
                return tensor.reshape(shape.value());
            }
        } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
            // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
            // is possible
            if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1] and
                tensor_shape.with_tile_padding()[-2] == shape.with_tile_padding()[-2]) {
                return tensor.reshape(shape.value());
            }
        }
    } else if (layout == ttnn::Layout::TILE) {
        const auto new_shape_with_tile_padding = shape.with_tile_padding();
        const auto new_height = new_shape_with_tile_padding[-2];
        const auto new_width = new_shape_with_tile_padding[-1];

        const auto is_tile_multiple = (new_height % ttnn::TILE_SIZE == 0 && new_width % ttnn::TILE_SIZE == 0);
        if (not is_tile_multiple) {
            TT_THROW(
                "Unable to reshape a tensor in TILE_LAYOUT to non-tile height and width! Please convert the tensor to "
                "ROW_MAJOR_LAYOUT first.");
        }

        if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
            if (tensor_shape.with_tile_padding()[-1] == new_width) {
                return tensor.reshape(shape.value());
            }
        } else {
            return tensor.reshape(shape.value());
        }
    }
    TT_THROW("Unable to reshape given tensor!");
}

template <std::size_t Rank>
inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const std::array<int32_t, Rank>& shape) {
    std::int64_t new_volume = 1;
    std::int64_t index_of_negative_1 = -1;
    for (auto index = 0; index < Rank; ++index) {
        if (shape[index] == -1) {
            if (index_of_negative_1 != -1) {
                TT_THROW("Shape cannot have more than 1 elements that is set to -1!");
            }
            index_of_negative_1 = index;
        }
        new_volume *= shape[index];
    }

    std::array<std::uint32_t, Rank> new_shape{};
    std::copy(shape.begin(), shape.end(), new_shape.begin());
    if (new_volume < 0) {
        const auto volume = tensor.get_shape().with_tile_padding().volume();
        new_shape[index_of_negative_1] = volume / (-new_volume);
    }
    return reshape(tensor, ttnn::Shape(new_shape));
}

inline ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    if (is_multi_device_tensor(tensor)) {
        return transform(tensor, [&](const Tensor& device_tensor) { return unsqueeze_to_4D(device_tensor); });
    }

    const auto tensor_shape = tensor.get_shape();
    const auto rank = tensor_shape.rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    const auto tensor_shape_4D = tensor_shape.to_rank<4>();
    return ttnn::operations::core::reshape(tensor, tensor_shape_4D);
}

inline ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, const int rank) {
    auto shape = tensor.get_shape();
    if (shape.rank() != 4) {
        TT_THROW("Tensor has to be of rank 4!");
    }
    if (rank < 1 or rank > 4) {
        TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
    }

    for (auto index = 0; index < 4 - rank; ++index) {
        if (shape[index] != 1) {
            TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
        }
    }

    switch (rank) {
        case 1: return ttnn::operations::core::reshape(tensor, shape.to_rank<1>());
        case 2: return ttnn::operations::core::reshape(tensor, shape.to_rank<2>());
        case 3: return ttnn::operations::core::reshape(tensor, shape.to_rank<3>());
        case 4: return tensor;
        default: TT_THROW("Invalid choice!");
    }
}

inline ttnn::Tensor to_memory_config(
    const ttnn::Tensor& tensor,
    const ttnn::MemoryConfig& memory_config,
    std::optional<ttnn::DataType> dtype = std::nullopt) {
    const auto original_memory_config = ttnn::get_memory_config(tensor);
    if (original_memory_config.has_value() && original_memory_config.value() == memory_config) {
        return tensor;
    }

    const auto original_shape = tensor.get_shape();
    const auto tensor_4D = unsqueeze_to_4D(tensor);

    if (memory_config.is_sharded()) {
        // to_sharded path
        if (tensor_4D.is_sharded()) {
            // reshard
            const auto input_memory_config = ttnn::get_memory_config(tensor_4D);
            const auto input_shard_spec = input_memory_config.value().shard_spec.value();
            const auto output_shard_spec = memory_config.shard_spec.value();
            if (tensor_4D.get_layout() == ttnn::TILE_LAYOUT ||
                input_shard_spec.shape[1] == output_shard_spec.shape[1]) {
                if (dtype.has_value()) {
                    throw runtime_error("dtype cannot be specified when converting sharded tensor to sharded tensor");
                }
                return reshape(tt::tt_metal::reshard(tensor_4D, memory_config), original_shape);
            } else {
                // for row-major tensors where shard-spec[1] is different for input shard and output shard
                return reshape(
                    tt::tt_metal::interleaved_to_sharded(
                        tt::tt_metal::sharded_to_interleaved(tensor_4D, ttnn::DRAM_MEMORY_CONFIG, dtype),
                        memory_config,
                        dtype),
                    original_shape);
            }
        } else {
            return reshape(tt::tt_metal::interleaved_to_sharded(tensor_4D, memory_config, dtype), original_shape);
        }
    } else {
        // to_interleaved path
        if (tensor_4D.is_sharded()) {
            return reshape(
                tt::tt_metal::sharded_to_interleaved(tensor_4D, memory_config, dtype), original_shape);
        } else {
            // L1 to DRAM or DRAM to L1
            return reshape(tt::tt_metal::clone(tensor_4D, memory_config, dtype), original_shape);
        }
    }

    return reshape(tensor_4D, original_shape);
}

inline ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking=true) { return tensor.cpu(blocking); }

inline Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& mem_config) {
    if (input_tensor.is_sharded()) {
        return move_sharded(input_tensor, mem_config);
    } else {
        return move(input_tensor, mem_config);
    }
}

inline Tensor to_layout(
    const ttnn::Tensor& tensor_arg,
    ttnn::Layout layout,
    std::optional<ttnn::DataType> dtype,
    std::optional<ttnn::MemoryConfig> memory_config) {
    if (tensor_arg.get_layout() == layout) {
        if (dtype.has_value() and dtype.value() != tensor_arg.get_dtype()) {
            tt::log_warning(
                tt::LogOp,
                "ttnn::to_layout: dtype is specified but the tensor is already in the requested layout! So, the dtype "
                "won't be changed!");
        }
        if (memory_config.has_value() and memory_config.value() != get_memory_config(tensor_arg).value()) {
            tt::log_warning(
                tt::LogOp,
                "ttnn::to_layout: memory_config is specified but the tensor is already in the requested layout! So, "
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
            bool use_multicore = tensor.is_sharded();
            if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
                return tt::tt_metal::untilize(tensor, output_memory_config, use_multicore);
            } else if (layout == ttnn::TILE_LAYOUT) {
                if (tensor.is_sharded()) {
                    const auto shard_shape = get_memory_config(tensor).value().shard_spec.value().shape;
                    if (shard_shape[0] % ttnn::TILE_SIZE != 0 or shard_shape[1] % ttnn::TILE_SIZE != 0) {
                        TT_THROW(
                            "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of TILE_SIZE!");
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
            return tensor.to(layout);

        } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            tensor = unsqueeze_to_4D(tensor);
            tensor = tensor.to(layout);
            tensor = tensor.unpad_from_tile(tensor.get_shape().value().without_padding());
            return reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape}));

        } else if (layout == ttnn::TILE_LAYOUT) {
            tensor = unsqueeze_to_4D(tensor);
            std::vector<uint32_t> padded_4D_output_shape;
            padded_4D_output_shape.push_back(tensor.get_shape()[-4]);
            padded_4D_output_shape.push_back(tensor.get_shape()[-3]);
            padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-2]));
            padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-1]));
            tensor = tensor.pad(padded_4D_output_shape, {0, 0, 0, 0}, 0).to(layout);
            return reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape, padded_output_shape}));

        } else {
            TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
        }
    }
}

}  // namespace core
}  // namespace operations

using operations::core::from_device;
using operations::core::reallocate;
using operations::core::reshape;
using operations::core::squeeze_from_4D;
using operations::core::to_layout;
using operations::core::unsqueeze_to_4D;


}  // namespace ttnn

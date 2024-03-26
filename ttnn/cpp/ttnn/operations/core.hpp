// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_eager/tt_numpy/functions.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/types.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace operations {
namespace core {

static inline const std::array<ttnn::TensorSchema, 1> reshape_input_schemas{
    ttnn::TensorSchema{
        1,
        8,
        {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16, ttnn::uint32, ttnn::float32},
        {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
        true,
        true,
        false,
        false},
};

inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    ttnn::validate_input_tensor("ttnn.reshape", tensor, reshape_input_schemas[0]);

    auto tensor_shape = tensor.get_shape();
    if (tensor_shape == shape) {
        return tensor;
    }

    const auto reshape_helper([](const ttnn::Tensor& tensor, const ttnn::Shape& shape) -> ttnn::Tensor {
        return tensor.reshape(shape.value());
    });

    //

    const auto layout = tensor.get_layout();

    if (layout == ttnn::Layout::ROW_MAJOR) {
        if (tensor.is_contiguous()) {
            if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
                // Page size depends on the width, so only modify the shape if the width is the same
                if (tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]) {
                    return reshape_helper(tensor, shape);
                }
            } else {
                return reshape_helper(tensor, shape);
            }
        } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
            // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
            // is possible
            if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1] and
                tensor_shape.with_tile_padding()[-2] == shape.with_tile_padding()[-2]) {
                return reshape_helper(tensor, shape);
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
                return reshape_helper(tensor, shape);
            }
        } else {
            return reshape_helper(tensor, shape);
        }
    }

    if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE) and tensor_shape.rank() == 4 and
        shape.rank() == 4 and tensor.get_dtype() == ttnn::bfloat16) {
        auto shape_with_tile_padding = shape.with_tile_padding();
        const auto w = shape_with_tile_padding[0];
        const auto z = shape_with_tile_padding[1];
        const auto y = shape_with_tile_padding[2];
        const auto x = shape_with_tile_padding[3];

        auto output_tensor = tt::tt_metal::reshape(tensor, w, z, y, x);
        return reshape_helper(output_tensor, shape);

    } else {
        TT_THROW("Unable to reshape given tensor!");
    }

    return tensor;
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

inline ttnn::Tensor from_device(const ttnn::Tensor& tensor) { return tensor.cpu(); }

// TODO : @eyonland move these creation functions to creation.hpp
template <typename T>
inline ttnn::Tensor full(
    const ttnn::Shape& shape,
    const T value,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return tt::numpy::full(shape.with_tile_padding().value(), value, data_type, layout, &device, memory_config);
}

inline ttnn::Tensor zeros(
    const ttnn::Shape& shape,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return full(shape, 0.0f, data_type, layout, device, memory_config);
}

inline ttnn::Tensor ones(
    const ttnn::Shape& shape,
    const DataType data_type,
    const Layout layout,
    Device& device,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return full(shape, 1.0f, data_type, layout, device, memory_config);
}

}  // namespace core
}  // namespace operations

using operations::core::from_device;
using operations::core::full;
using operations::core::ones;
using operations::core::zeros;
using operations::core::reshape;
using operations::core::unsqueeze_to_4D;
using operations::core::squeeze_from_4D;

}  // namespace ttnn

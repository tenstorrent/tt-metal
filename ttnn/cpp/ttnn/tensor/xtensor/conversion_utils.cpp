// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/assert.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>

namespace ttnn::experimental::xtensor {
namespace {

using ::tt::tt_metal::Tensor;

// copypaste from deprecated tensor pybinds ttnn
tt::tt_metal::OwnedBuffer create_owned_buffer(const std::vector<float>& data, DataType data_type) {
    using ::tt::tt_metal::owned_buffer::create;

    switch (data_type) {
        case DataType::BFLOAT8_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::BFLOAT4_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::FLOAT32: {
            auto data_copy = data;
            return create<float>(std::move(data_copy));
        }
        case DataType::BFLOAT16: {
            std::vector<bfloat16> bfloat16_data(data.size());
            std::transform(std::begin(data), std::end(data), std::begin(bfloat16_data), [](float value) {
                return bfloat16(value);
            });
            return create<bfloat16>(std::move(bfloat16_data));
        }
        default: {
            TT_THROW("Cannot create a host buffer for data type: {}", data_type);
        }
    }
}

template <typename T>
Tensor create_owned_tensor(
    std::vector<T> data, const ttnn::Shape& shape, tt::tt_metal::DataType data_type, tt::tt_metal::Layout layout) {
    auto buffer = tt::tt_metal::owned_buffer::create(std::move(data));
    auto storage = OwnedStorage{std::move(buffer)};
    return Tensor{std::move(storage), shape, data_type, layout};
}

// TODO: optimize precomputing multipliers
template <class T = float, class InternalT = bfloat16>
std::vector<T> untile_tensor_to_vec(const Tensor& cpu_tensor) {
    auto tiled_buffer = tt::tt_metal::host_buffer::get_as<InternalT>(cpu_tensor);
    auto untiled_shape = cpu_tensor.get_logical_shape();
    auto tiled_shape = cpu_tensor.get_padded_shape();

    // Calculate total size of the untiled tensor
    size_t total_size = untiled_shape.volume();

    std::vector<T> untiled_data(total_size);

    auto compute_flat_index = [](const std::vector<uint32_t>& indices, ttnn::SimpleShape& shape) -> uint32_t {
        uint32_t flat_index = 0;
        uint32_t multiplier = 1;
        for (int i = (int)indices.size() - 1; i >= 0; --i) {
            flat_index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return flat_index;
    };

    std::vector<uint32_t> indices(tiled_shape.rank(), 0);

    for (size_t idx = 0; idx < total_size; ++idx) {
        uint32_t untiled_index = compute_flat_index(indices, untiled_shape);
        uint32_t tiled_index = compute_flat_index(indices, tiled_shape);
        if constexpr (std::is_same_v<InternalT, bfloat16>) {
            untiled_data[untiled_index] = tiled_buffer[tiled_index].to_float();
        } else {
            untiled_data[untiled_index] = tiled_buffer[tiled_index];
        }

        for (int dim = (int)tiled_shape.rank() - 1; dim >= 0; --dim) {
            if (++indices[dim] < untiled_shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    return untiled_data;
}

}  // namespace

template <>
Tensor from_vector<float, DataType::BFLOAT16>(const std::vector<float>& buffer, const ttnn::Shape& shape) {
    const DataType data_type = DataType::BFLOAT16;
    auto logical_shape = shape.logical_shape();
    size_t volume = logical_shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    auto owned_buffer = create_owned_buffer(buffer, data_type);
    return Tensor(OwnedStorage{owned_buffer}, logical_shape, data_type, Layout::ROW_MAJOR);
}

// Workaround implementation due to issue with tilize for float32
// it is expected that tilize will be fixed in the after next tt-metal main update
template <>
Tensor from_vector<float, DataType::FLOAT32>(const std::vector<float>& buffer, const ttnn::Shape& shape) {
    auto tensor = from_vector<float, DataType::BFLOAT16>(buffer, shape);
    return ttnn::typecast(tensor, DataType::FLOAT32);
}

template <>
Tensor from_vector<uint32_t, DataType::UINT32>(const std::vector<uint32_t>& buffer, const ttnn::Shape& shape) {
    MemoryConfig output_mem_config{};
    auto logical_shape = shape.logical_shape();
    auto volume = logical_shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    return create_owned_tensor(buffer, logical_shape, DataType::UINT32, Layout::ROW_MAJOR);
}

template <>
Tensor from_vector<int32_t, DataType::INT32>(const std::vector<int32_t>& buffer, const ttnn::Shape& shape) {
    auto logical_shape = shape.logical_shape();
    auto volume = logical_shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    return create_owned_tensor(buffer, logical_shape, DataType::INT32, Layout::ROW_MAJOR);
}

template <>
std::vector<float> to_vector<float>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    if (cpu_tensor.get_dtype() == DataType::BFLOAT16) {
        return untile_tensor_to_vec<float, bfloat16>(cpu_tensor);
    } else if (cpu_tensor.get_dtype() == DataType::FLOAT32) {
        return untile_tensor_to_vec<float, float>(cpu_tensor);
    } else {
        TT_THROW("Cannot convert tensor to vector for data type: {}", cpu_tensor.get_dtype());
    }
}

template <>
std::vector<uint32_t> to_vector<uint32_t>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    return untile_tensor_to_vec<uint32_t, uint32_t>(cpu_tensor);
}

template <>
std::vector<int32_t> to_vector<int32_t>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    return untile_tensor_to_vec<int32_t, int32_t>(cpu_tensor);
}

}  // namespace ttnn::experimental::xtensor

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

using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Tensor;

template <typename T>
Tensor create_owned_tensor(
    tt::stl::Span<const T> data, const ttnn::SimpleShape& shape, DataType data_type, tt::tt_metal::Layout layout) {
    auto buffer = tt::tt_metal::owned_buffer::create(std::vector<T>(data.begin(), data.end()));
    auto storage = OwnedStorage{std::move(buffer)};
    return Tensor{std::move(storage), shape, data_type, layout};
}

// TODO: optimize precomputing multipliers
template <typename T, typename InternalType>
std::vector<T> untile_tensor_to_vec(const Tensor& cpu_tensor) {
    auto tiled_buffer = tt::tt_metal::host_buffer::get_as<InternalType>(cpu_tensor);
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
        if constexpr (std::is_same_v<InternalType, bfloat16>) {
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
Tensor from_span<float>(tt::stl::Span<const float> buffer, const ttnn::SimpleShape& shape, DataType dtype) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    if (dtype == DataType::FLOAT32) {
        return create_owned_tensor(buffer, shape, dtype, Layout::ROW_MAJOR);
    } else if (dtype == DataType::BFLOAT16) {
        std::vector<bfloat16> bfloat16_data;
        bfloat16_data.reserve(buffer.size());
        std::transform(std::begin(buffer), std::end(buffer), std::back_inserter(bfloat16_data), [](float value) {
            return bfloat16(value);
        });
        return create_owned_tensor(
            tt::stl::Span<const bfloat16>(bfloat16_data.data(), bfloat16_data.size()), shape, dtype, Layout::ROW_MAJOR);
    } else {
        // TODO: support bf8 and bf4
        TT_THROW("Unsupported data type for from_span<float>: {}", dtype);
    }
}

template <>
Tensor from_span<bfloat16>(tt::stl::Span<const bfloat16> buffer, const ttnn::SimpleShape& shape, DataType dtype) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(dtype == DataType::BFLOAT16, "Unsupported data type for from_span<bfloat16>: {}", dtype);
    return create_owned_tensor(buffer, shape, dtype, Layout::ROW_MAJOR);
}

template <>
Tensor from_span<uint32_t>(tt::stl::Span<const uint32_t> buffer, const ttnn::SimpleShape& shape, DataType dtype) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(dtype == DataType::UINT32, "Unsupported data type for from_span<uint32_t>: {}", dtype);
    return create_owned_tensor(buffer, shape, DataType::UINT32, Layout::ROW_MAJOR);
}

template <>
Tensor from_span<int32_t>(tt::stl::Span<const int32_t> buffer, const ttnn::SimpleShape& shape, DataType dtype) {
    size_t volume = shape.volume();
    TT_FATAL(
        buffer.size() == volume, "Current buffer size is {} different from shape volume {}", buffer.size(), volume);
    TT_FATAL(dtype == DataType::INT32, "Unsupported data type for from_span<int32_t>: {}", dtype);
    return create_owned_tensor(buffer, shape, DataType::INT32, Layout::ROW_MAJOR);
}

template <>
std::vector<float> to_vector<float>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    if (cpu_tensor.get_dtype() == DataType::BFLOAT16) {
        return untile_tensor_to_vec<float, bfloat16>(cpu_tensor);
    } else if (cpu_tensor.get_dtype() == DataType::FLOAT32) {
        return untile_tensor_to_vec<float, float>(cpu_tensor);
    } else {
        // TODO: support bf4, bf8.
        TT_THROW("Cannot convert tensor to vector for data type: {}", cpu_tensor.get_dtype());
    }
}

template <>
std::vector<bfloat16> to_vector<bfloat16>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    TT_FATAL(
        cpu_tensor.get_dtype() == DataType::BFLOAT16,
        "Unsupported data type for to_vector<bfloat16>: {}",
        cpu_tensor.get_dtype());
    return untile_tensor_to_vec<bfloat16, bfloat16>(cpu_tensor);
}

template <>
std::vector<uint32_t> to_vector<uint32_t>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    TT_FATAL(
        cpu_tensor.get_dtype() == DataType::UINT32,
        "Unsupported data type for to_vector<uint32_t>: {}",
        cpu_tensor.get_dtype());
    return untile_tensor_to_vec<uint32_t, uint32_t>(cpu_tensor);
}

template <>
std::vector<int32_t> to_vector<int32_t>(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().to(Layout::ROW_MAJOR);
    TT_FATAL(
        cpu_tensor.get_dtype() == DataType::INT32,
        "Unsupported data type for to_vector<int32_t>: {}",
        cpu_tensor.get_dtype());
    return untile_tensor_to_vec<int32_t, int32_t>(cpu_tensor);
}

}  // namespace ttnn::experimental::xtensor

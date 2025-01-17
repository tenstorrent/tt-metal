// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/math.hpp>
#include <optional>
#include <random>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/host_buffer/types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/tensor/tensor_impl.hpp>
#include "cpp/ttnn/common/constants.hpp"

namespace ttnn {

using tt::tt_metal::DataType;
using tt::tt_metal::IDevice;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::OwnedStorage;
using tt::tt_metal::StorageType;
using tt::tt_metal::Tensor;

template <typename T, bool IS_UPPER>
static Tensor index_trilu(
    const tt::tt_metal::LegacyShape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    // Current implementation restrictions
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));

    auto index = 0;
    auto rank = shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    auto offset = shape[penultimate] * shape[ultimate];
    auto iterations = 1;
    for (int itr = 0; itr < rank - 2; itr++) {
        iterations *= shape[itr];
    }
    for (uint32_t itr = 0; itr < iterations; itr++) {
        for (int32_t y = 0; y < shape[penultimate]; y++) {
            for (int32_t x = 0; x < shape[ultimate]; x++) {
                int32_t value = (IS_UPPER) ? (x >= (y + diag)) : (y >= (x - diag));
                if constexpr (std::is_same_v<T, ::bfloat16>) {
                    owned_buffer[index + y * shape[ultimate] + x] = T(static_cast<float>(value));
                } else {
                    owned_buffer[index + y * shape[ultimate] + x] = static_cast<T>(value);
                }
            }  // dim X
        }  // dim Y
        index += offset;
    }
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_width(
    const tt::tt_metal::LegacyShape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto& up_shape = shape.without_padding();
    auto index = 0;
    auto value = 0;
    auto rank = up_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < up_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < up_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < up_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < up_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
                value = 0;
                index = index + (shape[ultimate] - up_shape[ultimate]);
            }  // dim H
            index = index + ((shape[penultimate] - up_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim c
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_height(
    const tt::tt_metal::LegacyShape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto& up_shape = shape.without_padding();
    auto index = 0;
    auto value = 0;
    auto rank = up_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < up_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < up_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < up_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < up_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                value = value + 1;
                index = index + (shape[ultimate] - up_shape[ultimate]);
            }  // dim H
            value = 0;
            index = index + ((shape[penultimate] - up_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_all(
    const tt::tt_metal::LegacyShape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto& up_shape = shape.without_padding();
    auto index = 0;
    auto value = 0;
    auto rank = up_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < up_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < up_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < up_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < up_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
                index = index + (shape[ultimate] - up_shape[ultimate]);
            }  // dim H
            index = index + ((shape[penultimate] - up_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor mask_padded_input(
    const tt::tt_metal::LegacyShape& padded_shape,
    const tt::tt_metal::LegacyShape& unpadded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(padded_shape));

    auto index = 0;
    auto rank = padded_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < padded_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < padded_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < padded_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < padded_shape[ultimate]; x++) {
                    if (b < unpadded_shape[rank - 4] && c < unpadded_shape[rank - 3] &&
                        y < unpadded_shape[penultimate] && x < unpadded_shape[ultimate]) {
                        owned_buffer[index++] = T(static_cast<float>(1.0));
                    } else {
                        owned_buffer[index++] = T(static_cast<float>(0.0));
                    }
                }  // dim W
            }  // dim H
        }  // dim C
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, padded_shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor fill_first_val_into_tensor(
    const Tensor& input_tensor,
    DataType data_type,
    const Layout layout,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto physical_volume = input_tensor.volume();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(physical_volume);  // ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
    const ttnn::SimpleShape input_tensor_strides = input_tensor.strides();
    for (uint32_t i = 0; i < physical_volume; i++) {
        owned_buffer[i] = input_buffer[0];
    }
    const tt::tt_metal::LegacyShape& s_a = input_tensor.get_legacy_shape();
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor prod_result_computation_GS(
    const Tensor& input_tensor,
    DataType data_type,
    const Layout layout,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const tt::tt_metal::LegacyShape& s_a = input_tensor.get_legacy_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(input_tensor.volume());  // ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
    const ttnn::SimpleShape input_tensor_strides = input_tensor.strides();
    auto result = static_cast<T>(1.0f);
    for (uint32_t i = s_a[0] - 1; i < s_a[0]; i++) {
        for (int32_t j = s_a[1] - 1; j < s_a[1]; j++) {
            for (int32_t k = s_a[2] - 32; k < s_a[2]; k++) {  // access last tile
                for (int32_t l = s_a[3] - 32; l < s_a[3]; l++) {
                    auto input_index =
                        l + input_tensor_strides[2] * k + input_tensor_strides[1] * j + input_tensor_strides[0] * i;
                    if (k >= s_a[2] - 2 && l >= s_a[3] - 32) {  // to access 2*32 in TILE layout
                        result = result * static_cast<T>(input_buffer[input_index]);
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    } else {
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    }
                }
            }
        }
    }
    owned_buffer[0] = result;  // store the result at the first position of the tensor,and the rest of the values as
                               // 0.0f
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor prod_result_computation_WH_B0(
    const Tensor& input_tensor,
    DataType data_type,
    const Layout layout,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const tt::tt_metal::LegacyShape& s_a = input_tensor.get_legacy_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(s_a));  // ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
    const ttnn::SimpleShape input_tensor_strides = input_tensor.strides();
    auto result = static_cast<T>(1.0f);
    // need to access the last 4 rows and alternating columns of index 17 ,19, 21, 23, 25, 27, 29, 31
    for (uint32_t i = s_a[0] - 1; i < s_a[0]; i++) {
        for (int32_t j = s_a[1] - 1; j < s_a[1]; j++) {
            for (int32_t k = s_a[2] - 32; k < s_a[2]; k++) {  // access last tile
                for (int32_t l = s_a[3] - 32; l < s_a[3]; l++) {
                    auto input_index =
                        l + input_tensor_strides[2] * k + input_tensor_strides[1] * j + input_tensor_strides[0] * i;
                    if (k >= s_a[2] - 4 && (l == s_a[3] - 15 || l == s_a[3] - 13 || l == s_a[3] - 11 ||
                                            l == s_a[3] - 9 || l == s_a[3] - 7 || l == s_a[3] - 5 || l == s_a[3] - 3 ||
                                            l == s_a[3] - 1)) {  // to access 4*16 elements placed alternatively
                                                                 // starting from index 17W in TILE layout
                        result = result * static_cast<T>(input_buffer[input_index]);
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    } else {
                        owned_buffer[input_index] = static_cast<T>(0.0f);
                    }
                }
            }
        }
    }
    owned_buffer[0] = result;  // store the result at the first position of the tensor,and the rest of the values as
                               // 0.0f
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_channel(
    const tt::tt_metal::LegacyShape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto& up_shape = shape.without_padding();
    auto index = 0;
    auto value = 0;
    auto rank = up_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < up_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < up_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < up_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < up_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                index = index + (shape[ultimate] - up_shape[ultimate]);
            }  // dim H
            value = value + 1;
            index = index + ((shape[penultimate] - up_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
        value = 0;
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_batch(
    const tt::tt_metal::LegacyShape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto& up_shape = shape.without_padding();
    auto index = 0;
    auto value = 0;
    auto rank = up_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < up_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < up_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < up_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < up_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                index = index + (shape[ultimate] - up_shape[ultimate]);
            }  // dim H
            index = index + ((shape[penultimate] - up_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
        value = value + 1;
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor manual_insertion(
    const Tensor& input_tensor,
    const tt::tt_metal::LegacyShape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        shape[0] * shape[1] * shape[2] * shape[3] == input_tensor.volume(),
        "Required shape volume must match old shape volume");
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto owned_buffer = owned_buffer::create<T>(std::move(data_vec));
    auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_tril(
    const tt::tt_metal::LegacyShape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, false>(shape, diag, data_type, layout, device, output_mem_config);
}

template <typename T>
static Tensor index_triu(
    const tt::tt_metal::LegacyShape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, true>(shape, diag, data_type, layout, device, output_mem_config);
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

static void seed(std::size_t seed) { RANDOM_GENERATOR = std::mt19937(seed); }

template <typename T>
static Tensor uniform(T low, T high, const ttnn::SimpleShape& shape, const Layout layout = Layout::ROW_MAJOR) {
    constexpr DataType data_type = tt::tt_metal::convert_to_data_type<T>();

    TensorSpec spec(shape, TensorLayout(data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(spec.padded_shape().volume());

    if constexpr (std::is_same_v<T, uint32_t>) {
        auto rand_value = std::bind(std::uniform_int_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < owned_buffer.size(); index++) {
            owned_buffer[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, float>) {
        auto rand_value = std::bind(std::uniform_real_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < owned_buffer.size(); index++) {
            owned_buffer[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, ::bfloat16>) {
        auto rand_value =
            std::bind(std::uniform_real_distribution<float>(low.to_float(), high.to_float()), RANDOM_GENERATOR);
        for (auto index = 0; index < owned_buffer.size(); index++) {
            owned_buffer[index] = ::bfloat16(rand_value());
        }
    }

    return Tensor(OwnedStorage{owned_buffer}, spec).to(layout);
}

static Tensor random(
    const ttnn::SimpleShape& shape,
    const DataType data_type = DataType::BFLOAT16,
    const Layout layout = Layout::ROW_MAJOR) {
    switch (data_type) {
        case DataType::UINT8: return uniform(uint8_t(0), uint8_t(1), shape, layout);
        case DataType::UINT16: return uniform(uint16_t(0), uint16_t(1), shape, layout);
        case DataType::UINT32: return uniform(0u, 1u, shape, layout);
        case DataType::FLOAT32: return uniform(0.0f, 1.0f, shape, layout);
        case DataType::BFLOAT16: return uniform(::bfloat16(0.0f), ::bfloat16(1.0f), shape, layout);
        default: TT_THROW("Unsupported DataType!");
    };
}

}  // namespace random

namespace detail {
static bool nearly_equal(float a, float b, float epsilon = 1e-5f, float abs_threshold = 1e-5f) {
    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
    auto result = diff < std::max(abs_threshold, epsilon * norm);
    if (not result) {
        tt::log_error(tt::LogTest, "{} != {}", a, b);
    }
    return result;
}

template <typename... Args>
static bool nearly_equal(::bfloat16 a, ::bfloat16 b, Args... args) {
    return nearly_equal(a.to_float(), b.to_float(), args...);
}
}  // namespace detail

template <typename DataType, typename... Args>
static bool allclose(const Tensor& tensor_a, const Tensor& tensor_b, Args... args) {
    if (tensor_a.get_legacy_shape() != tensor_b.get_legacy_shape()) {
        return false;
    }

    if (tensor_a.get_dtype() != tensor_b.get_dtype()) {
        return false;
    }

    auto tensor_a_buffer = tt::tt_metal::owned_buffer::get_as<DataType>(tensor_a);
    auto tensor_b_buffer = tt::tt_metal::owned_buffer::get_as<DataType>(tensor_b);

    for (int index = 0; index < tensor_a_buffer.size(); index++) {
        using ::ttnn::detail::nearly_equal;
        if (not nearly_equal(tensor_a_buffer[index], tensor_b_buffer[index], args...)) {
            return false;
        }
    }
    return true;
}

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/math.hpp>
#include <tt-metalium/overloaded.hpp>
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
using tt::tt_metal::PageConfig;
using tt::tt_metal::StorageType;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;

namespace detail {
template <typename T>
tt::tt_metal::owned_buffer::Buffer<T> to_host_buffer(const Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto& storage = cpu_tensor.storage();
    tt::tt_metal::OwnedBuffer buffer = std::visit(
        tt::stl::overloaded{
            [](const tt::tt_metal::OwnedStorage& storage) { return storage.get_buffer(); },
            [](const tt::tt_metal::MultiDeviceHostStorage& storage) {
                TT_FATAL(storage.num_buffers() == 1, "Can't get a single buffer from multi device host storage");
                return storage.get_buffer(0);
            },
            [](const auto&) -> tt::tt_metal::OwnedBuffer { TT_THROW("Not supported storage type"); }},
        storage);
    return std::get<tt::tt_metal::owned_buffer::Buffer<T>>(buffer);
}
}  // namespace detail

template <typename T, bool IS_UPPER>
static Tensor index_trilu(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    // Current implementation restrictions
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());

    auto index = 0;
    auto rank = padded_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    auto offset = padded_shape[penultimate] * padded_shape[ultimate];
    auto iterations = 1;
    for (int itr = 0; itr < rank - 2; itr++) {
        iterations *= padded_shape[itr];
    }
    for (uint32_t itr = 0; itr < iterations; itr++) {
        for (int32_t y = 0; y < padded_shape[penultimate]; y++) {
            for (int32_t x = 0; x < padded_shape[ultimate]; x++) {
                int32_t value = (IS_UPPER) ? (x >= (y + diag)) : (y >= (x - diag));
                if constexpr (std::is_same_v<T, ::bfloat16>) {
                    owned_buffer[index + y * padded_shape[ultimate] + x] = T(static_cast<float>(value));
                } else {
                    owned_buffer[index + y * padded_shape[ultimate] + x] = static_cast<T>(value);
                }
            }  // dim X
        }  // dim Y
        index += offset;
    }
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_width(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
                value = 0;
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim c
    }  // dim N
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_height(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                value = value + 1;
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            value = 0;
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
    }  // dim N
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_all(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
    }  // dim N
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor mask_padded_input(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());

    auto index = 0;
    auto rank = padded_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < padded_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < padded_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < padded_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < padded_shape[ultimate]; x++) {
                    if (b < logical_shape[rank - 4] && c < logical_shape[rank - 3] && y < logical_shape[penultimate] &&
                        x < logical_shape[ultimate]) {
                        owned_buffer[index++] = T(static_cast<float>(1.0));
                    } else {
                        owned_buffer[index++] = T(static_cast<float>(0.0));
                    }
                }  // dim W
            }  // dim H
        }  // dim C
    }  // dim N
    auto output = Tensor(OwnedStorage{owned_buffer}, padded_shape, data_type, Layout::ROW_MAJOR).to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
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
    auto input_buffer = detail::to_host_buffer<T>(input_tensor);
    const ttnn::Shape input_tensor_strides = input_tensor.strides();
    for (uint32_t i = 0; i < physical_volume; i++) {
        owned_buffer[i] = input_buffer[0];
    }
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          input_tensor.get_logical_shape(),
                          TensorLayout::fromPaddedShape(
                              data_type,
                              PageConfig(Layout::ROW_MAJOR),
                              MemoryConfig{},
                              input_tensor.get_logical_shape(),
                              input_tensor.get_padded_shape())))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
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
    const ttnn::Shape& s_a = input_tensor.get_padded_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(input_tensor.volume());  // ouput
    auto input_buffer = detail::to_host_buffer<T>(input_tensor);
    const ttnn::Shape input_tensor_strides = input_tensor.strides();
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
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          input_tensor.get_logical_shape(),
                          TensorLayout::fromPaddedShape(
                              data_type,
                              Layout::ROW_MAJOR,
                              MemoryConfig{},
                              input_tensor.get_logical_shape(),
                              input_tensor.get_padded_shape())))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
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
    const auto& s_a = input_tensor.get_padded_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(s_a.volume());  // ouput
    auto input_buffer = detail::to_host_buffer<T>(input_tensor);
    const ttnn::Shape input_tensor_strides = input_tensor.strides();
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
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          input_tensor.get_logical_shape(),
                          TensorLayout::fromPaddedShape(
                              data_type,
                              PageConfig(Layout::ROW_MAJOR),
                              MemoryConfig{},
                              input_tensor.get_logical_shape(),
                              input_tensor.get_padded_shape())))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_channel(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            value = value + 1;
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
        value = 0;
    }  // dim N
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_batch(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(padded_shape.volume());
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    owned_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
        value = value + 1;
    }  // dim N
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor manual_insertion(
    const Tensor& input_tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        padded_shape[0] * padded_shape[1] * padded_shape[2] * padded_shape[3] == input_tensor.volume(),
        "Required shape volume must match old shape volume");
    auto owned_buffer = detail::to_host_buffer<T>(input_tensor);
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      TensorSpec(
                          logical_shape,
                          TensorLayout::fromPaddedShape(
                              data_type, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)))
                      .to_layout(layout);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_tril(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, false>(logical_shape, padded_shape, diag, data_type, layout, device, output_mem_config);
}

template <typename T>
static Tensor index_triu(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, true>(logical_shape, padded_shape, diag, data_type, layout, device, output_mem_config);
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

static void seed(std::size_t seed) { RANDOM_GENERATOR = std::mt19937(seed); }

template <typename T>
static Tensor uniform(T low, T high, const ttnn::Shape& shape, const Layout layout = Layout::ROW_MAJOR) {
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

    return Tensor(OwnedStorage{owned_buffer}, spec).to_layout(layout);
}

static Tensor random(
    const ttnn::Shape& shape, const DataType data_type = DataType::BFLOAT16, const Layout layout = Layout::ROW_MAJOR) {
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
    if (tensor_a.get_padded_shape() != tensor_b.get_padded_shape()) {
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

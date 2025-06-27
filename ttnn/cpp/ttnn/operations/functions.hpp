// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iterator>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt_stl/overloaded.hpp>
#include <optional>
#include <random>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/tensor/tensor_impl.hpp>
#include "ttnn/common/constants.hpp"

namespace ttnn {

using tt::tt_metal::DataType;
using tt::tt_metal::IDevice;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::StorageType;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;

template <typename T, bool IS_UPPER>
static Tensor index_trilu(
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    IDevice* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    // Current implementation restrictions
    auto output_buffer = std::vector<T>(padded_shape.volume());

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
                    output_buffer[index + y * padded_shape[ultimate] + x] = T(static_cast<float>(value));
                } else {
                    output_buffer[index + y * padded_shape[ultimate] + x] = static_cast<T>(value);
                }
            }  // dim X
        }  // dim Y
        index += offset;
    }
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(padded_shape.volume());
    std::fill(output_buffer.begin(), output_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    output_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
                value = 0;
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim c
    }  // dim N
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(padded_shape.volume());
    std::fill(output_buffer.begin(), output_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    output_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                value = value + 1;
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            value = 0;
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
    }  // dim N
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(padded_shape.volume());
    std::fill(output_buffer.begin(), output_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    output_buffer[index++] = T(static_cast<float>(value));
                    value = value + 1;
                }  // dim W
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
    }  // dim N
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(padded_shape.volume());

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
                        output_buffer[index++] = T(static_cast<float>(1.0));
                    } else {
                        output_buffer[index++] = T(static_cast<float>(0.0));
                    }
                }  // dim W
            }  // dim H
        }  // dim C
    }  // dim N
    auto output = Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), padded_shape, data_type, Layout::ROW_MAJOR)
                      .to_layout(layout);
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto physical_volume = input_tensor.physical_volume();
    auto output_buffer = std::vector<T>(physical_volume);
    auto input_cpu_tensor = input_tensor.cpu();
    tt::stl::Span<const T> host_buffer = tt::tt_metal::host_buffer::get_as<T>(input_cpu_tensor);
    const ttnn::Shape input_tensor_strides = input_tensor.strides();
    for (uint32_t i = 0; i < physical_volume; i++) {
        output_buffer[i] = host_buffer[0];
    }
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
                      TensorSpec(
                          input_tensor.logical_shape(),
                          TensorLayout::fromPaddedShape(
                              data_type,
                              PageConfig(Layout::ROW_MAJOR),
                              MemoryConfig{},
                              input_tensor.logical_shape(),
                              input_tensor.padded_shape())))
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(tt::constants::TILE_HW);
    auto input_cpu_tensor = input_tensor.cpu();
    tt::stl::Span<const T> input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_cpu_tensor);
    const ttnn::Shape input_tensor_strides = input_tensor.strides();
    T result = static_cast<T>(1.0f);

    // Calculate the product of all elements in the last tile
    for (int i = 0; i < tt::constants::TILE_HW; ++i) {
        result = result * static_cast<T>(input_buffer[i]);
    }
    output_buffer[0] = result;
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
                      TensorSpec(
                          ttnn::Shape({}),
                          TensorLayout::fromPaddedShape(
                              data_type,
                              PageConfig(Layout::ROW_MAJOR),
                              MemoryConfig{},
                              /*logical_shape=*/ttnn::Shape({}),
                              /*padded_shape=*/ttnn::Shape({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}))))

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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(padded_shape.volume());
    std::fill(output_buffer.begin(), output_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    output_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            value = value + 1;
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
        value = 0;
    }  // dim N
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    auto output_buffer = std::vector<T>(padded_shape.volume());
    std::fill(output_buffer.begin(), output_buffer.end(), -std::numeric_limits<float>::infinity());
    auto index = 0;
    auto value = 0;
    auto rank = logical_shape.rank();
    auto penultimate = rank - 2;
    auto ultimate = rank - 1;
    for (uint32_t b = 0; b < logical_shape[rank - 4]; b++) {
        for (uint32_t c = 0; c < logical_shape[rank - 3]; c++) {
            for (uint32_t y = 0; y < logical_shape[penultimate]; y++) {
                for (uint32_t x = 0; x < logical_shape[ultimate]; x++) {
                    output_buffer[index++] = T(static_cast<float>(value));
                }  // dim W
                index = index + (padded_shape[ultimate] - logical_shape[ultimate]);
            }  // dim H
            index = index + ((padded_shape[penultimate] - logical_shape[penultimate]) * tt::constants::TILE_WIDTH);
        }  // dim C
        value = value + 1;
    }  // dim N
    auto output = Tensor(
                      tt::tt_metal::HostBuffer(std::move(output_buffer)),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    TT_ASSERT(input_tensor.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        padded_shape[0] * padded_shape[1] * padded_shape[2] * padded_shape[3] == input_tensor.physical_volume(),
        "Required shape volume must match old shape volume");
    auto input_cpu_tensor = input_tensor.cpu();
    auto output = Tensor(
                      tt::tt_metal::host_buffer::get_host_buffer(input_cpu_tensor),
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
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
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    return index_trilu<T, true>(logical_shape, padded_shape, diag, data_type, layout, device, output_mem_config);
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

// the effect of instantiating a template uniform_int_distribution is undefined
// unless IntType satisfy the following concept according to 26.6.2.1 General requirements [rand.req.genl]
template <typename T>
concept IntType =
    std::is_same_v<T, short> || std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned short> || std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, unsigned long long>;

inline void seed(std::size_t seed) { RANDOM_GENERATOR = std::mt19937(seed); }

template <typename T>
static Tensor uniform(T low, T high, const ttnn::Shape& shape, const Layout layout = Layout::ROW_MAJOR) {
    constexpr DataType data_type = tt::tt_metal::convert_to_data_type<T>();

    TensorSpec spec(shape, TensorLayout(data_type, PageConfig(layout), MemoryConfig{}));
    auto output_buffer = std::vector<T>(spec.padded_shape().volume());

    if constexpr (std::is_same_v<T, uint32_t>) {
        auto rand_value = std::bind(std::uniform_int_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < output_buffer.size(); index++) {
            output_buffer[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, float>) {
        auto rand_value = std::bind(std::uniform_real_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < output_buffer.size(); index++) {
            output_buffer[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, ::bfloat16>) {
        auto rand_value =
            std::bind(std::uniform_real_distribution<float>(low.to_float(), high.to_float()), RANDOM_GENERATOR);
        for (auto index = 0; index < output_buffer.size(); index++) {
            output_buffer[index] = ::bfloat16(rand_value());
        }
    }

    return Tensor(tt::tt_metal::HostBuffer(std::move(output_buffer)), spec);
}

inline Tensor random(
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
inline bool nearly_equal(float a, float b, float epsilon = 1e-5f, float abs_threshold = 1e-5f) {
    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
    auto result = diff < std::max(abs_threshold, epsilon * norm);
    if (not result) {
        log_error(tt::LogTest, "{} != {}", a, b);
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
    if (tensor_a.padded_shape() != tensor_b.padded_shape()) {
        return false;
    }

    if (tensor_a.dtype() != tensor_b.dtype()) {
        return false;
    }

    tt::stl::Span<const DataType> tensor_a_buffer = tt::tt_metal::host_buffer::get_as<DataType>(tensor_a);
    tt::stl::Span<const DataType> tensor_b_buffer = tt::tt_metal::host_buffer::get_as<DataType>(tensor_b);

    for (int index = 0; index < tensor_a_buffer.size(); index++) {
        using ::ttnn::detail::nearly_equal;
        if (not nearly_equal(tensor_a_buffer[index], tensor_b_buffer[index], args...)) {
            return false;
        }
    }
    return true;
}

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <common/math.hpp>
#include <optional>
#include <random>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/host_buffer/types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/tensor/tensor_impl.hpp>
#include "ttnn/cpp/ttnn/common/constants.hpp"

namespace ttnn {

namespace numpy {

using tt::tt_metal::DataType;
using tt::tt_metal::Device;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::OwnedStorage;
using tt::tt_metal::StorageType;
using tt::tt_metal::Tensor;
namespace detail {

template <typename T>
constexpr static DataType get_data_type() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DataType::UINT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, ::bfloat16>) {
        return DataType::BFLOAT16;
    } else {
        TT_THROW("Unsupported DataType!");
    }
}

template <typename T>
static Tensor full(
    uint8_t queue_id,
    const ttnn::Shape& shape,
    T value,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    if (layout == Layout::TILE) {
        if (shape.rank() < 2) {
            TT_THROW("TILE layout requires rank >= 2");
        }
        TT_FATAL(
                shape[-1] % tt::constants::TILE_WIDTH == 0,
                "TILE layout requires width dimension to be multiple of 32");

        TT_FATAL(
                shape[-2] % tt::constants::TILE_HEIGHT == 0,
                "TILE layout requires height dimension to be multiple of 32");
    }

        constexpr DataType data_type = detail::get_data_type<T>();
        auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
        std::fill(std::begin(owned_buffer), std::end(owned_buffer), value);

        if(!optional_output_tensor.has_value()){
            auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
            if (device != nullptr) {
                output = output.to(device, output_mem_config);
            }
            return output;
        }
        else {
            auto device_buffer = std::get<DeviceStorage>(optional_output_tensor.value().tensor_attributes->storage).get_buffer();
            bool using_fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);

            if (using_fast_dispatch && device != nullptr) {
                auto& cmd_queue = device->command_queue(queue_id);
                if (CommandQueue::default_mode() == CommandQueue::CommandQueueMode::ASYNC) {
                    tt::tt_metal::EnqueueWriteBuffer(cmd_queue, device_buffer, owned_buffer.get_ptr(), false);
                } else {
                    tt::tt_metal::EnqueueWriteBuffer(cmd_queue, device_buffer, owned_buffer.data(), false);
                }
            } else {
                auto uint32_data = tt::tt_metal::tensor_impl::pack_vec_into_uint32_vec<T>(owned_buffer);
                tt::tt_metal::detail::WriteToBuffer(*device_buffer, uint32_data);
            }

            return optional_output_tensor.value();
        }
}

}  // namespace detail

template <typename T>
static Tensor full_impl(
    uint8_t queue_id,
    const ttnn::Shape& shape,
    const T value,
    const DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    switch (data_type) {
        case DataType::UINT8: {
            return detail::full<uint8_t>(queue_id, shape, uint8_t(value), layout, device, output_mem_config, optional_output_tensor);
        }
        case DataType::UINT16: {
            return detail::full<uint16_t>(queue_id, shape, uint16_t(value), layout, device, output_mem_config, optional_output_tensor);
        }
        case DataType::UINT32: {
            return detail::full<uint32_t>(queue_id, shape, uint32_t(value), layout, device, output_mem_config, optional_output_tensor);
        }
        case DataType::FLOAT32: {
            return detail::full<float>(queue_id, shape, float(value), layout, device, output_mem_config, optional_output_tensor);
        }
        case DataType::BFLOAT16: {
            return detail::full<::bfloat16>(
                queue_id, shape, ::bfloat16(static_cast<float>(value)), layout, device, output_mem_config, optional_output_tensor);
        }
        default: TT_THROW("Unsupported DataType!");
    }
}

template <typename T>
static Tensor full(
    const ttnn::Shape& shape,
    const T value,
    const DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full_impl(ttnn::DefaultQueueId, shape, value, data_type, layout, device, output_mem_config, std::nullopt);
}

static Tensor zeros(
    const ttnn::Shape& shape,
    const DataType data_type = DataType::BFLOAT16,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full(shape, 0.0f, data_type, layout, device, output_mem_config);
}

static Tensor ones(
    const ttnn::Shape& shape,
    const DataType data_type = DataType::BFLOAT16,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return full(shape, 1.0f, data_type, layout, device, output_mem_config);
}

template <typename T>
static Tensor full_like(
    const Tensor& input_tensor,
    const T value,
    std::optional<DataType> data_type = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    DataType data_type_to_use = input_tensor.get_dtype();
    if (data_type.has_value()) {
        data_type_to_use = data_type.value();
    }
    Layout layout_to_use = input_tensor.get_layout();
    if (layout.has_value()) {
        layout_to_use = layout.value();
    }
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        MemoryConfig output_mem_config_to_use = input_tensor.memory_config();
        if (output_mem_config.has_value()) {
            output_mem_config_to_use = output_mem_config.value();
        }
        return full(
            input_tensor.get_shape(),
            value,
            data_type_to_use,
            layout_to_use,
            input_tensor.device(),
            output_mem_config_to_use);
    } else {
        return full(input_tensor.get_shape(), value, data_type_to_use, layout_to_use);
    }
}

static Tensor zeros_like(
    const Tensor& input_tensor,
    std::optional<DataType> data_type = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    return full_like(input_tensor, 0.0f, data_type, layout, output_mem_config);
}

static Tensor ones_like(
    const Tensor& input_tensor,
    std::optional<DataType> data_type = std::nullopt,
    std::optional<Layout> layout = std::nullopt,
    std::optional<MemoryConfig> output_mem_config = std::nullopt) {
    return full_like(input_tensor, 1.0f, data_type, layout, output_mem_config);
}

template <typename T>
static Tensor arange(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    constexpr DataType data_type = detail::get_data_type<T>();
    // Current implementation restrictions
    TT_ASSERT(step > 0, "Step must be greater than 0");
    TT_ASSERT(start < stop, "Start must be less than step");
    auto size = tt::div_up((stop - start), step);
    if (size % 2 != 0) {
        size++;
    }
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(size);

    auto index = 0;
    for (auto value = start; value < stop; value += step) {
        if constexpr (std::is_same_v<T, ::bfloat16>) {
            owned_buffer[index++] = T(static_cast<float>(value));
        } else {
            owned_buffer[index++] = static_cast<T>(value);
        }
    }
    auto output = Tensor(OwnedStorage{owned_buffer}, {1, 1, 1, static_cast<uint32_t>(size)}, data_type, layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T, bool IS_UPPER>
static Tensor index_trilu(
    const ttnn::Shape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
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
    for (int itr = 0; itr < rank - 2; itr++) iterations *= shape[itr];
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
    const ttnn::Shape& up_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto& shape = up_shape.with_tile_padding();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
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
    const ttnn::Shape& up_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto& shape = up_shape.with_tile_padding();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
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
    const ttnn::Shape& up_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto& shape = up_shape.with_tile_padding();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
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
    const ttnn::Shape& padded_shape,
    const ttnn::Shape& unpadded_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
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
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const ttnn::Shape& s_a = input_tensor.get_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(s_a));  // ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
    const ttnn::Shape input_tensor_strides = input_tensor.strides();
    for (uint32_t i = 0; i < tt::tt_metal::compute_volume(s_a); i++) {
        owned_buffer[i] = input_buffer[0];
    }
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, layout).to(layout);
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
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const ttnn::Shape& s_a = input_tensor.get_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(s_a));  // ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
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
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, layout).to(layout);
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
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    const ttnn::Shape& s_a = input_tensor.get_shape();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(s_a));  // ouput
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
            input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
    } else {
        tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto input_buffer = owned_buffer::create<T>(std::move(data_vec));
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
    auto output = Tensor(OwnedStorage{owned_buffer}, s_a, data_type, layout).to(layout);
    if (device != nullptr) {
        output = output.to(device, output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor index_channel(
    const ttnn::Shape& up_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto& shape = up_shape.with_tile_padding();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
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
    const ttnn::Shape& up_shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    auto& shape = up_shape.with_tile_padding();
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));
    std::fill(owned_buffer.begin(), owned_buffer.end(), -std::numeric_limits<float>::infinity());
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
    const ttnn::Shape& shape,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        shape[0] * shape[1] * shape[2] * shape[3] == input_tensor.volume(),
        "Required shape volume must match old shape volume");
    auto device_buffer = input_tensor.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
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
    const ttnn::Shape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, false>(shape, diag, data_type, layout, device, output_mem_config);
}

template <typename T>
static Tensor index_triu(
    const ttnn::Shape& shape,
    const int32_t diag,
    DataType data_type,
    const Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    return index_trilu<T, true>(shape, diag, data_type, layout, device, output_mem_config);
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

static void seed(std::size_t seed) { RANDOM_GENERATOR = std::mt19937(seed); }

template <typename T>
static Tensor uniform(T low, T high, const ttnn::Shape& shape, const Layout layout = Layout::ROW_MAJOR) {
    constexpr DataType data_type = detail::get_data_type<T>();

    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(tt::tt_metal::compute_volume(shape));

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

    return Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
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
    if (tensor_a.get_shape() != tensor_b.get_shape()) {
        return false;
    }

    if (tensor_a.get_dtype() != tensor_b.get_dtype()) {
        return false;
    }

    auto tensor_a_buffer = tt::tt_metal::owned_buffer::get_as<DataType>(tensor_a);
    auto tensor_b_buffer = tt::tt_metal::owned_buffer::get_as<DataType>(tensor_b);

    for (int index = 0; index < tensor_a_buffer.size(); index++) {
        if (not detail::nearly_equal(tensor_a_buffer[index], tensor_b_buffer[index], args...)) {
            return false;
        }
    }
    return true;
}

}  // namespace numpy
}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/deprecated/tt_numpy/functions.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace creation {

template <typename T, std::size_t rank=4>
Tensor create_scalar(T scalar, DataType data_type, Layout layout, Device* device){
    static_assert(rank >=2, "Rank must be at least 2 when creating a tensor with TILE_LAYOUT");
    std::array<std::uint32_t, rank> intended_shape = {};
    intended_shape.fill(1);
    std::array<std::uint32_t, rank> device_shape = {};
    device_shape.fill(1);

    if(layout == Layout::ROW_MAJOR){
        device_shape[device_shape.size() - 2] = 2;
        auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(2));
        host_buffer[0] = scalar;
        Tensor scalar_tensor_host = Tensor(
            OwnedStorage{host_buffer},
            ttnn::Shape(intended_shape, device_shape),
            data_type,
            Layout::ROW_MAJOR);
        return scalar_tensor_host.to(device);
    }
    else if(layout == Layout::TILE){
        device_shape[device_shape.size() - 2] = TILE_HEIGHT;
        device_shape[device_shape.size() - 1] = TILE_WIDTH;
        auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(TILE_HEIGHT * TILE_WIDTH));
        host_buffer[0] = scalar;
        Tensor scalar_tensor_host = Tensor(
            OwnedStorage{host_buffer},
            ttnn::Shape(intended_shape, device_shape),
            data_type,
            Layout::TILE);
        return scalar_tensor_host.to(device);
    }
    else{
        throw std::runtime_error("Unsupported layout");
    }
}

template <typename T>
inline ttnn::Tensor full(
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    Device* device = device_arg.has_value() ? &(device_arg.value().get()) : nullptr;
    return tt::numpy::full(
        shape.value,
        fill_value,
        dtype.value_or(ttnn::bfloat16),
        layout.value_or(ttnn::ROW_MAJOR_LAYOUT),
        device,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

inline ttnn::Tensor zeros(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return full(shape, 0.0f, dtype, layout, device, memory_config);
}

inline ttnn::Tensor ones(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return full(shape, 1.0f, dtype, layout, device, memory_config);
}

inline ttnn::Tensor empty(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return full(shape, 0.0f, dtype, layout, device, memory_config);
}

template <typename T>
inline ttnn::Tensor full_like(
    const ttnn::Tensor& tensor,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    if (ttnn::is_tensor_on_device_or_multidevice(tensor)) {
        return full(
            tensor.get_shape(),
            fill_value,
            dtype.value_or(tensor.get_dtype()),
            layout.value_or(tensor.get_layout()),
            device.value_or(*tensor.device()),
            memory_config.value_or(tensor.memory_config()));
    } else {
        return full(
            tensor.get_shape(),
            fill_value,
            dtype.value_or(tensor.get_dtype()),
            layout.value_or(tensor.get_layout()),
            device,
            memory_config);
    }
}

inline ttnn::Tensor zeros_like(
    const ttnn::Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return full_like(tensor, 0.0f, dtype, layout, device, memory_config);
}

inline ttnn::Tensor ones_like(
    const ttnn::Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return full_like(tensor, 1.0f, dtype, layout, device, memory_config);
}

inline ttnn::Tensor empty_like(
    const ttnn::Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return full_like(tensor, 0.0f, dtype, layout, device, memory_config);
}

struct Full {
    static ttnn::Tensor operator()(
        const ttnn::Shape& shape,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full(shape, fill_value, dtype, layout, device, memory_config);
    }

    static ttnn::Tensor operator()(
        const ttnn::Shape& shape,
        const int fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full(shape, fill_value, dtype, layout, device, memory_config);
    }
};

struct FullLike {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& tensor,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full_like(tensor, fill_value, dtype, layout, device, memory_config);
    }

    static ttnn::Tensor operator()(
        const ttnn::Tensor& tensor,
        const int fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full_like(tensor, fill_value, dtype, layout, device, memory_config);
    }
};

struct Arange {
    static ttnn::Tensor operator()(
        const int64_t stop,
        const DataType dtype = ttnn::bfloat16,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        return Arange::operator()(0, stop, 1, dtype, device, memory_config);
    }

    static ttnn::Tensor operator()(
        const int64_t start,
        const int64_t stop,
        const int64_t step = 1,
        const DataType dtype = ttnn::bfloat16,
        const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        Device* device = device_arg.has_value() ? &(device_arg.value().get()) : nullptr;
        switch (dtype) {
            case ttnn::bfloat16:
                return tt::numpy::arange<::bfloat16>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::float32:
                return tt::numpy::arange<float>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::uint16:
                return tt::numpy::arange<uint16_t>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::uint32:
                return tt::numpy::arange<uint32_t>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::int32:
                return tt::numpy::arange<int32_t>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            default: TT_THROW("Unsupported dtype");
        }
    }
};

}  // namespace creation
}  // namespace operations

constexpr auto full = ttnn::register_operation_with_auto_launch_op<"ttnn::full", ttnn::operations::creation::Full>();
constexpr auto zeros = REGISTER_OPERATION_FROM_FUNCTION("ttnn::zeros", ttnn::operations::creation::zeros);
constexpr auto ones = REGISTER_OPERATION_FROM_FUNCTION("ttnn::ones", ttnn::operations::creation::ones);
constexpr auto empty = REGISTER_OPERATION_FROM_FUNCTION("ttnn::empty", ttnn::operations::creation::empty);

constexpr auto full_like = ttnn::register_operation_with_auto_launch_op<"ttnn::full_like", ttnn::operations::creation::FullLike>();
constexpr auto zeros_like =
    REGISTER_OPERATION_FROM_FUNCTION("ttnn::zeros_like", ttnn::operations::creation::zeros_like);
constexpr auto ones_like = REGISTER_OPERATION_FROM_FUNCTION("ttnn::ones_like", ttnn::operations::creation::ones_like);
constexpr auto empty_like =
    REGISTER_OPERATION_FROM_FUNCTION("ttnn::empty_like", ttnn::operations::creation::empty_like);

constexpr auto arange = ttnn::register_operation_with_auto_launch_op<"ttnn::arange", ttnn::operations::creation::Arange>();

}  // namespace ttnn

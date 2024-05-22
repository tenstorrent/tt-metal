// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_numpy/functions.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace operations {
namespace creation {

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
        shape.value(),
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
    static ttnn::Tensor execute(
        const ttnn::Shape& shape,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full(shape, fill_value, dtype, layout, device, memory_config);
    }

    static ttnn::Tensor execute(
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
    static ttnn::Tensor execute(
        const ttnn::Tensor& tensor,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full_like(tensor, fill_value, dtype, layout, device, memory_config);
    }

    static ttnn::Tensor execute(
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
    static ttnn::Tensor execute(
        const int64_t stop,
        const DataType dtype = ttnn::bfloat16,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        return Arange::execute(0, stop, 1, dtype, device, memory_config);
    }

    static ttnn::Tensor execute(
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

constexpr auto full = ttnn::register_operation<ttnn::operations::creation::Full>("ttnn::full");
constexpr auto zeros = ttnn::register_operation<ttnn::operations::creation::zeros>("ttnn::zeros");
constexpr auto ones = ttnn::register_operation<ttnn::operations::creation::ones>("ttnn::ones");
constexpr auto empty = ttnn::register_operation<ttnn::operations::creation::empty>("ttnn::empty");

constexpr auto full_like = ttnn::register_operation<ttnn::operations::creation::FullLike>("ttnn::full_like");
constexpr auto zeros_like = ttnn::register_operation<ttnn::operations::creation::zeros_like>("ttnn::zeros_like");
constexpr auto ones_like = ttnn::register_operation<ttnn::operations::creation::ones_like>("ttnn::ones_like");
constexpr auto empty_like = ttnn::register_operation<ttnn::operations::creation::empty_like>("ttnn::empty_like");

constexpr auto arange = ttnn::register_operation<ttnn::operations::creation::Arange>("ttnn::arange");

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

namespace ttnn {
namespace operations {
namespace creation {

template <typename T, std::size_t rank=4>
Tensor create_scalar(T scalar, DataType data_type, Layout layout, Device* device){
    using namespace tt::constants;
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
inline ttnn::Tensor full_impl(
    uint8_t queue_id,
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
    Device* device = optional_output_tensor.has_value() ? optional_output_tensor.value().device() : device_arg.has_value() ? &(device_arg.value().get()) : nullptr;
    Layout layout_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_layout() : layout.value_or(ttnn::ROW_MAJOR_LAYOUT);
    DataType dtype_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_dtype() : dtype.value_or(ttnn::bfloat16);
    tt::tt_metal::LegacyShape shape_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_legacy_shape() : shape.value;
    MemoryConfig mem_cfg = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return numpy::full_impl(
        queue_id,
        shape_value,
        fill_value,
        dtype_value,
        layout_value,
        device,
        mem_cfg,
        optional_output_tensor);
}

template <typename T>
inline ttnn::Tensor full(
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    uint8_t queue_id = ttnn::DefaultQueueId) {
    return full_impl(queue_id, shape, fill_value, dtype, layout, device_arg, memory_config, optional_output_tensor);
}

namespace detail {

// Non-type template parameters (NTTPs) disallow floating point values
// This works around that limitation by using a structural type
// https://godbolt.org/z/hxKje3MYe
template <class T>
struct boxed {
    T value;
    consteval boxed(T value) noexcept : value(value) {}
    consteval auto invoke() const noexcept -> T { return value; }
};

} // namespace detail

template <detail::boxed FillValue>
struct FullWith {
    static constexpr auto fill_value = FillValue.invoke();

    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full(shape, fill_value, dtype, layout, device, memory_config);
    }
};

struct Zeros : FullWith<0.0f> {};
struct Ones : FullWith<1.0f> {};

inline constexpr Zeros zeros{};
inline constexpr Ones ones{};

template <typename T>
inline ttnn::Tensor full_like_impl(
    uint8_t queue_id,
    const ttnn::Tensor& tensor,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
    if (ttnn::is_tensor_on_device_or_multidevice(tensor)) {
        return full_impl(
            queue_id,
            tensor.get_shape(),
            fill_value,
            dtype.value_or(tensor.get_dtype()),
            layout.value_or(tensor.get_layout()),
            device.value_or(*tensor.device()),
            memory_config.value_or(tensor.memory_config()),
            optional_output_tensor);
    } else {
        return full_impl(
            queue_id,
            tensor.get_shape(),
            fill_value,
            dtype.value_or(tensor.get_dtype()),
            layout.value_or(tensor.get_layout()),
            device,
            memory_config,
            optional_output_tensor);
    }
}

template <typename T>
inline ttnn::Tensor full_like(
    const ttnn::Tensor& tensor,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {

    return full_like_impl(ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, std::nullopt);
}

template <detail::boxed FillValue>
struct FullLikeWith {
    static constexpr auto fill_value = FillValue.invoke();

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct OnesLike : FullLikeWith<1.0f> {};

inline constexpr OnesLike ones_like{};

struct Empty {
   static ttnn::Tensor invoke(
    const ttnn::Shape& shape,
    const DataType& dtype,
    const Layout& layout,
    Device* device,
    const MemoryConfig& memory_config) {
        return create_device_tensor(shape, dtype, layout, device, memory_config);
    }
};

struct EmptyLike {
   static ttnn::Tensor invoke(
    const ttnn::Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    Device* device = device_arg.has_value() ? &(device_arg.value().get()) : tensor.device();
    Layout layout_value = layout.value_or(tensor.get_layout());
    DataType dtype_value = dtype.value_or(tensor.get_dtype());
    MemoryConfig mem_cfg = memory_config.value_or(tensor.memory_config());
        return create_device_tensor(tensor.get_shape(), dtype_value, layout_value, device, mem_cfg);
    }
};


struct ZerosLike {
   static ttnn::Tensor invoke(
    uint8_t queue_id,
    const ttnn::Tensor& tensor,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt ) {

        if(!optional_output_tensor.has_value()) {
            Device* device = device_arg.has_value() ? &(device_arg.value().get()) : tensor.device();
            Layout layout_value = layout.value_or(tensor.get_layout());
            DataType dtype_value = dtype.value_or(tensor.get_dtype());
            MemoryConfig mem_cfg = memory_config.value_or(tensor.memory_config());
            optional_output_tensor = create_device_tensor(tensor.get_shape(), dtype_value, layout_value, device, mem_cfg);
        }

        // this if() {...} can be skipped if RM support is not needed for zeros_like
        if(optional_output_tensor.value().get_layout() == Layout::ROW_MAJOR) {
            Tensor x = optional_output_tensor.value();
            x = ttnn::to_layout(x, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, x.device());
            ttnn::fill(x, 0, std::nullopt, x);
            x = ttnn::to_layout(x, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, x.device());
            ttnn::assign(x, optional_output_tensor.value());
            return optional_output_tensor.value();
        }

        ttnn::fill(optional_output_tensor.value(), 0, std::nullopt, optional_output_tensor);
        return optional_output_tensor.value();
    }

    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return invoke(ttnn::DefaultQueueId, tensor, dtype, layout, device, memory_config, optional_output_tensor);
    }

};


struct Full {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Shape& shape,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(queue_id, shape, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Shape& shape,
        const int fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(queue_id, shape, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(ttnn::DefaultQueueId, shape, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const int fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(ttnn::DefaultQueueId, shape, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct FullLike {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& tensor,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& tensor,
        const int fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const float fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const int fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct Arange {
    static ttnn::Tensor invoke(
        const int64_t stop,
        const DataType dtype = ttnn::bfloat16,
        const std::optional<std::reference_wrapper<Device>>& device = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        return Arange::invoke(0, stop, 1, dtype, device, memory_config);
    }

    static ttnn::Tensor invoke(
        const int64_t start,
        const int64_t stop,
        const int64_t step = 1,
        const DataType dtype = ttnn::bfloat16,
        const std::optional<std::reference_wrapper<Device>>& device_arg = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        Device* device = device_arg.has_value() ? &(device_arg.value().get()) : nullptr;
        switch (dtype) {
            case ttnn::bfloat16:
                return numpy::arange<::bfloat16>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::float32:
                return numpy::arange<float>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::uint16:
                return numpy::arange<uint16_t>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::uint32:
                return numpy::arange<uint32_t>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            case ttnn::int32:
                return numpy::arange<int32_t>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
            default: TT_THROW("Unsupported dtype");
        }
    }
};

}  // namespace creation
}  // namespace operations

constexpr auto full =
    ttnn::decorators::register_operation_with_auto_launch_op<"ttnn::full", ttnn::operations::creation::Full>();
constexpr auto zeros = ttnn::decorators::register_operation<"ttnn::zeros", ttnn::operations::creation::Zeros>();
constexpr auto ones = ttnn::decorators::register_operation<"ttnn::ones", ttnn::operations::creation::Ones>();
constexpr auto empty = ttnn::decorators::register_operation<"ttnn::empty", ttnn::operations::creation::Empty>();

constexpr auto full_like =
    ttnn::decorators::register_operation_with_auto_launch_op<"ttnn::full_like", ttnn::operations::creation::FullLike>();
constexpr auto zeros_like =
    ttnn::decorators::register_operation<"ttnn::zeros_like", ttnn::operations::creation::ZerosLike>();
constexpr auto ones_like =
    ttnn::decorators::register_operation<"ttnn::ones_like", ttnn::operations::creation::OnesLike>();
constexpr auto empty_like =
    ttnn::decorators::register_operation<"ttnn::empty_like", ttnn::operations::creation::EmptyLike>();

constexpr auto arange =
    ttnn::decorators::register_operation_with_auto_launch_op<"ttnn::arange", ttnn::operations::creation::Arange>();

}  // namespace ttnn

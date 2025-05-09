// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <variant>

#include <tt-metalium/command_queue.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace creation {

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

template <typename T>
Tensor arange_impl(
    const int64_t start,
    const int64_t stop,

    const int64_t step,
    const Layout layout = Layout::ROW_MAJOR,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const MemoryConfig& output_mem_config = MemoryConfig{}) {
    constexpr DataType data_type = tt::tt_metal::convert_to_data_type<T>();

    TT_FATAL(step != 0, "Step must be nonzero");
    TT_FATAL(
        !((step > 0 && start > stop) || (step < 0 && start < stop)),
        "Invalid range: Step direction does not match range bounds");

    auto size = std::max<int64_t>(0, tt::div_up(std::abs(stop - start), std::abs(step)));
    auto owned_buffer = std::vector<T>(size);

    auto index = 0;
    for (auto value = start; (step > 0) ? (value < stop) : (value > stop); value += step) {
        if constexpr (std::is_same_v<T, ::bfloat16>) {
            owned_buffer[index++] = T(static_cast<float>(value));
        } else {
            owned_buffer[index++] = static_cast<T>(value);
        }
    }

    auto output = Tensor(
                      tt::tt_metal::HostStorage{tt::tt_metal::host_buffer::create(std::move(owned_buffer))},
                      ttnn::Shape{static_cast<uint32_t>(size)},
                      data_type,
                      Layout::ROW_MAJOR)
                      .to_layout(layout);
    if (device.has_value()) {
        return output.to_device(&device->get(), output_mem_config);
    }
    return output;
}

template <typename T>
Tensor full_impl(
    QueueId queue_id,
    const ttnn::Shape& shape,
    T value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor) {
    constexpr DataType data_type = tt::tt_metal::convert_to_data_type<T>();
    TensorSpec tensor_spec(shape, TensorLayout(data_type, PageConfig(layout), MemoryConfig{}));
    auto owned_buffer = std::vector<T>(tensor_spec.physical_shape().height() * tensor_spec.physical_shape().width());
    std::fill(std::begin(owned_buffer), std::end(owned_buffer), value);

    Tensor host_tensor(
        tt::tt_metal::HostStorage{tt::tt_metal::host_buffer::create(std::move(owned_buffer))},
        shape,
        data_type,
        layout);

    if (optional_output_tensor.has_value()) {
        tt::tt_metal::write_tensor(host_tensor, *optional_output_tensor, queue_id);
        return *optional_output_tensor;
    } else if (device != nullptr) {
        return host_tensor.to_device(device, output_mem_config);
    } else {
        return host_tensor;
    }
}

}  // namespace detail

template <typename T>
Tensor full_impl(
    QueueId queue_id,
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    MeshDevice* device = nullptr,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    MeshDevice* device_to_use = optional_output_tensor.has_value() ? optional_output_tensor->mesh_device() : device;

    DataType dtype_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_dtype()
                                                              : dtype.value_or(DataType::BFLOAT16);
    auto get_default_layout = [dtype_value]() {
        return (dtype_value == DataType::BFLOAT4_B || dtype_value == DataType::BFLOAT8_B) ? ttnn::TILE_LAYOUT
                                                                                          : ttnn::ROW_MAJOR_LAYOUT;
    };

    Layout layout_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_layout()
                                                             : layout.value_or(get_default_layout());
    ttnn::Shape shape_value =
        optional_output_tensor.has_value() ? optional_output_tensor.value().get_logical_shape() : shape;
    MemoryConfig mem_cfg = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                              : memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    auto concrete_full = [&]<typename BufferType>(BufferType fill_value) {
        return detail::full_impl<BufferType>(
            queue_id, shape_value, fill_value, layout_value, device_to_use, mem_cfg, optional_output_tensor);
    };

    switch (dtype_value) {
        case DataType::UINT8: return concrete_full.template operator()<uint8_t>(fill_value);
        case DataType::UINT16: return concrete_full.template operator()<uint16_t>(fill_value);
        case DataType::UINT32: return concrete_full.template operator()<uint32_t>(fill_value);
        case DataType::FLOAT32: return concrete_full.template operator()<float>(fill_value);
        case DataType::BFLOAT16: return concrete_full.template operator()<::bfloat16>(static_cast<float>(fill_value));
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B: {
            TensorSpec tensor_spec(shape_value, TensorLayout(dtype_value, PageConfig(layout_value), mem_cfg));
            std::vector<float> fill_value_vec(shape_value.volume(), static_cast<float>(fill_value));
            auto output = tt::tt_metal::Tensor::from_vector(std::move(fill_value_vec), tensor_spec);
            if (device_to_use != nullptr) {
                output = output.to_device(device_to_use, mem_cfg);
            }
            return output;
        }
        default: TT_THROW("Unsupported DataType!");
    }
}

template <detail::boxed FillValue>
struct FullWith {
    static constexpr auto fill_value = FillValue.invoke();

    static Tensor invoke(
        const ttnn::Shape& shape,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return full_impl(
            ttnn::DefaultQueueId,
            shape,
            fill_value,
            dtype,
            layout,
            device.has_value() ? &device->get() : nullptr,
            memory_config,
            std::nullopt);
    }
};

struct Zeros : FullWith<0.0f> {};
struct Ones : FullWith<1.0f> {};

inline constexpr Zeros zeros{};
inline constexpr Ones ones{};

template <typename T>
Tensor full_like_impl(
    QueueId queue_id,
    const Tensor& tensor,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<std::reference_wrapper<MeshDevice>> device_arg = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt) {
    MeshDevice* device = device_arg.has_value() ? &device_arg->get() : nullptr;
    Layout layout_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_layout()
                                                             : layout.value_or(tensor.get_layout());
    DataType dtype_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_dtype()
                                                              : dtype.value_or(tensor.get_dtype());
    auto arch = tensor.device()->arch();
    const bool is_tile_layout = (tensor.get_layout() == Layout::TILE) && (layout_value == Layout::TILE);
    if (tt::tt_metal::is_device_tensor(tensor)) {
        // requires reference tensor to be in TILE for device operation fill - this will be changed later
        if (is_tile_layout &&
            (dtype_value == DataType::BFLOAT8_B || dtype_value == DataType::BFLOAT16 ||
             (arch != tt::ARCH::GRAYSKULL && dtype_value == DataType::FLOAT32)) &&
            tensor.storage_type() == StorageType::DEVICE) {
            return ttnn::fill(tensor, fill_value, memory_config, optional_output_tensor);
        } else {
            return full_impl(
                queue_id,
                tensor.get_logical_shape(),
                fill_value,
                dtype_value,
                layout_value,
                device ? device : tensor.mesh_device(),
                memory_config.value_or(tensor.memory_config()),
                optional_output_tensor);
        }
    } else {
        return full_impl(
            queue_id,
            tensor.get_logical_shape(),
            fill_value,
            dtype_value,
            layout_value,
            device ? device : tensor.mesh_device(),
            memory_config,
            optional_output_tensor);
    }
}

template <detail::boxed FillValue>
struct FullLikeWith {
    static constexpr auto fill_value = FillValue.invoke();

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static Tensor invoke(
        const Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct ZerosLike : FullLikeWith<0.0f> {};
struct OnesLike : FullLikeWith<1.0f> {};

inline constexpr ZerosLike zeros_like{};
inline constexpr OnesLike ones_like{};

struct Empty {
    static Tensor invoke(
        const ttnn::Shape& shape,
        const DataType& dtype,
        const Layout& layout,
        MeshDevice* device,
        const MemoryConfig& memory_config) {
        return allocate_tensor_on_mesh(
            TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config)), device);
    }
};

struct EmptyLike {
    static Tensor invoke(
        const Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        Layout layout_value = layout.value_or(tensor.get_layout());
        DataType dtype_value = dtype.value_or(tensor.get_dtype());
        MemoryConfig mem_cfg = memory_config.value_or(tensor.memory_config());
        return allocate_tensor_on_mesh(
            TensorSpec(tensor.get_logical_shape(), TensorLayout(dtype_value, PageConfig(layout_value), mem_cfg)),
            device.has_value() ? &device->get() : tensor.mesh_device());
    }
};

struct Full {
    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static Tensor invoke(
        QueueId queue_id,
        const ttnn::Shape& shape,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(
            queue_id,
            shape,
            fill_value,
            dtype,
            layout,
            device.has_value() ? &device->get() : nullptr,
            memory_config,
            optional_output_tensor);
    }

    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static Tensor invoke(
        const ttnn::Shape& shape,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(
            ttnn::DefaultQueueId,
            shape,
            fill_value,
            dtype,
            layout,
            device.has_value() ? &device->get() : nullptr,
            memory_config,
            optional_output_tensor);
    }
};

struct FullLike {
    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& tensor,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static Tensor invoke(
        const Tensor& tensor,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct Arange {
    static Tensor invoke(
        const int64_t stop,
        const DataType dtype = DataType::BFLOAT16,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        return Arange::invoke(0, stop, 1, dtype, device, memory_config);
    }

    static Tensor invoke(
        const int64_t start,
        const int64_t stop,
        const int64_t step = 1,
        const DataType dtype = ttnn::DataType::BFLOAT16,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        auto concrete_arange = [&]<typename BufferType>() {
            return detail::arange_impl<BufferType>(start, stop, step, ttnn::ROW_MAJOR_LAYOUT, device, memory_config);
        };

        switch (dtype) {
            case DataType::BFLOAT16: return concrete_arange.template operator()<::bfloat16>();
            case DataType::FLOAT32: return concrete_arange.template operator()<float>();
            case DataType::UINT16: return concrete_arange.template operator()<uint16_t>();
            case DataType::UINT32: return concrete_arange.template operator()<uint32_t>();
            case DataType::INT32: return concrete_arange.template operator()<int32_t>();
            default: TT_THROW("Unsupported dtype");
        }
    }
};

}  // namespace creation
}  // namespace operations

constexpr auto full = ttnn::decorators::register_operation<"ttnn::full", ttnn::operations::creation::Full>();
constexpr auto zeros = ttnn::decorators::register_operation<"ttnn::zeros", ttnn::operations::creation::Zeros>();
constexpr auto ones = ttnn::decorators::register_operation<"ttnn::ones", ttnn::operations::creation::Ones>();
constexpr auto empty = ttnn::decorators::register_operation<"ttnn::empty", ttnn::operations::creation::Empty>();

constexpr auto full_like =
    ttnn::decorators::register_operation<"ttnn::full_like", ttnn::operations::creation::FullLike>();
constexpr auto zeros_like =
    ttnn::decorators::register_operation<"ttnn::zeros_like", ttnn::operations::creation::ZerosLike>();
constexpr auto ones_like =
    ttnn::decorators::register_operation<"ttnn::ones_like", ttnn::operations::creation::OnesLike>();
constexpr auto empty_like =
    ttnn::decorators::register_operation<"ttnn::empty_like", ttnn::operations::creation::EmptyLike>();

constexpr auto arange = ttnn::decorators::register_operation<"ttnn::arange", ttnn::operations::creation::Arange>();

}  // namespace ttnn

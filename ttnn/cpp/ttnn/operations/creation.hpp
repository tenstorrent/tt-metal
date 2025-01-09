// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <variant>

#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/any_device.hpp"
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

// Helper class to transparently bind instances of Device / MeshDevice along with their reference wrappers to
// AnyDevice
class OptionalAnyDevice {
public:
    OptionalAnyDevice() = default;
    OptionalAnyDevice(std::nullopt_t);
    OptionalAnyDevice(ttnn::AnyDevice device);
    OptionalAnyDevice(const std::optional<std::reference_wrapper<tt::tt_metal::IDevice>>& device);
    OptionalAnyDevice(const std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice>>& mesh_device);
    OptionalAnyDevice(std::reference_wrapper<tt::tt_metal::IDevice> device);
    OptionalAnyDevice(std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> mesh_device);
    OptionalAnyDevice(tt::tt_metal::IDevice& device);
    OptionalAnyDevice(tt::tt_metal::distributed::MeshDevice& mesh_device);

    OptionalAnyDevice(const OptionalAnyDevice&) = default;
    OptionalAnyDevice& operator=(const OptionalAnyDevice&) = default;
    OptionalAnyDevice(OptionalAnyDevice&&) = delete;
    OptionalAnyDevice& operator=(OptionalAnyDevice&&) = delete;

    bool has_value() { return device_.has_value(); }
    ttnn::AnyDevice* operator->() { return &(*device_); }
    ttnn::AnyDevice operator*() { return *device_; }

private:
    std::optional<ttnn::AnyDevice> device_;
};

// Converts an instance of AnyDevice to a vector of the underlying Devices.
// TODO: Consider moving the helper into a dedicated header with the related utils.
inline std::vector<IDevice*> get_workers_from_device(OptionalAnyDevice device) {
    return device.has_value() ? device->get_devices() : std::vector<IDevice*>{};
}

template <typename T>
static Tensor arange_impl(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout = Layout::ROW_MAJOR,
    OptionalAnyDevice device = std::nullopt,
    const MemoryConfig& output_mem_config = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) {
    constexpr DataType data_type = tt::tt_metal::convert_to_data_type<T>();
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
    auto output = Tensor(
                      OwnedStorage{owned_buffer},
                      ttnn::SimpleShape{1, 1, 1, static_cast<uint32_t>(size)},
                      data_type,
                      Layout::ROW_MAJOR)
                      .to(layout);
    if (device.has_value()) {
        output = output.to(device->get_devices(), output_mem_config);
    }
    return output;
}

template <typename T>
static Tensor full_impl(
    uint8_t queue_id,
    const tt::tt_metal::LegacyShape& shape,
    T value,
    const Layout layout,
    const std::vector<IDevice*>& devices,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor) {
    constexpr DataType data_type = tt::tt_metal::convert_to_data_type<T>();
    TensorSpec tensor_spec(
        shape.logical_shape(),
        TensorLayout::fromLegacyPaddedShape(data_type, PageConfig(layout), MemoryConfig{}, shape));
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(
        tensor_spec.physical_shape().height() * tensor_spec.physical_shape().width());
    // TODO: 15061 - Generalize the header to support generic vector / view types.
    std::fill(std::begin(owned_buffer), std::end(owned_buffer), value);

    if (!optional_output_tensor.has_value()) {
        auto output = Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
        if (!devices.empty()) {
            output = output.to(devices, output_mem_config);
        }
        return output;
    } else {
        const auto buffers = optional_output_tensor->buffers();
        const bool using_fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);

        for (auto* buffer : buffers) {
            if (using_fast_dispatch) {
                auto& cmd_queue = buffer->device()->command_queue(queue_id);
                if (CommandQueue::default_mode() == CommandQueue::CommandQueueMode::ASYNC) {
                    tt::tt_metal::EnqueueWriteBuffer(cmd_queue, *buffer, owned_buffer.get_ptr(), /*blocking=*/false);
                } else {
                    tt::tt_metal::EnqueueWriteBuffer(cmd_queue, *buffer, owned_buffer.data(), /*blocking=*/false);
                }
            } else {
                tt::tt_metal::detail::WriteToBuffer(*buffer, owned_buffer.get());
            }
        }

        return *optional_output_tensor;
    }
}

}  // namespace detail

template <typename T, std::size_t rank = 4>
Tensor create_scalar(T scalar, DataType data_type, Layout layout, IDevice* device) {
    using namespace tt::constants;
    static_assert(rank >= 2, "Rank must be at least 2 when creating a tensor with TILE_LAYOUT");
    std::array<std::uint32_t, rank> intended_shape = {};
    intended_shape.fill(1);
    std::array<std::uint32_t, rank> device_shape = {};
    device_shape.fill(1);

    if (layout == Layout::ROW_MAJOR) {
        device_shape[device_shape.size() - 2] = 2;
        auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(2));
        host_buffer[0] = scalar;
        Tensor scalar_tensor_host =
            Tensor(OwnedStorage{host_buffer}, ttnn::Shape(intended_shape, device_shape), data_type, Layout::ROW_MAJOR);
        return scalar_tensor_host.to(device);
    } else if (layout == Layout::TILE) {
        device_shape[device_shape.size() - 2] = TILE_HEIGHT;
        device_shape[device_shape.size() - 1] = TILE_WIDTH;
        auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(TILE_HEIGHT * TILE_WIDTH));
        host_buffer[0] = scalar;
        Tensor scalar_tensor_host =
            Tensor(OwnedStorage{host_buffer}, ttnn::Shape(intended_shape, device_shape), data_type, Layout::TILE);
        return scalar_tensor_host.to(device);
    } else {
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
    const std::vector<IDevice*>& workers = {},
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
    const std::vector<IDevice*>& workers_to_use =
        optional_output_tensor.has_value() ? optional_output_tensor->get_workers(/*blocking=*/true) : workers;

    Layout layout_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_layout()
                                                             : layout.value_or(ttnn::ROW_MAJOR_LAYOUT);
    DataType dtype_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_dtype()
                                                              : dtype.value_or(DataType::BFLOAT16);
    tt::tt_metal::LegacyShape shape_value =
        optional_output_tensor.has_value() ? optional_output_tensor.value().get_legacy_shape() : shape.value;
    MemoryConfig mem_cfg = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                              : memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    auto concrete_full = [&]<typename BufferType>(BufferType fill_value) {
        return detail::full_impl<BufferType>(
            queue_id, shape_value, fill_value, layout_value, workers, mem_cfg, optional_output_tensor);
    };

    switch (dtype_value) {
        case DataType::UINT8: return concrete_full.template operator()<uint8_t>(fill_value);
        case DataType::UINT16: return concrete_full.template operator()<uint16_t>(fill_value);
        case DataType::UINT32: return concrete_full.template operator()<uint32_t>(fill_value);
        case DataType::FLOAT32: return concrete_full.template operator()<float>(fill_value);
        case DataType::BFLOAT16: return concrete_full.template operator()<::bfloat16>(static_cast<float>(fill_value));
        default: TT_THROW("Unsupported DataType!");
    }
}

template <typename T>
inline ttnn::Tensor full(
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    detail::OptionalAnyDevice device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    uint8_t queue_id = ttnn::DefaultQueueId) {
    return full_impl(
        queue_id,
        shape,
        fill_value,
        dtype,
        layout,
        detail::get_workers_from_device(device),
        memory_config,
        optional_output_tensor);
}

template <detail::boxed FillValue>
struct FullWith {
    static constexpr auto fill_value = FillValue.invoke();

    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device = std::nullopt,
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
    detail::OptionalAnyDevice device = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
    Layout layout_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_layout()
                                                             : layout.value_or(tensor.get_layout());
    DataType dtype_value = optional_output_tensor.has_value() ? optional_output_tensor.value().get_dtype()
                                                              : dtype.value_or(tensor.get_dtype());
    auto arch = tensor.device()->arch();
    bool is_TILE = (tensor.get_layout() == Layout::TILE) && (layout_value == Layout::TILE);
    if (ttnn::is_tensor_on_device_or_multidevice(tensor)) {
        // requires reference tensor to be in TILE for device operation fill - this will be changed later
        if (is_TILE &&
            (dtype_value == DataType::BFLOAT8_B || dtype_value == DataType::BFLOAT16 ||
             (arch != tt::ARCH::GRAYSKULL && dtype_value == DataType::FLOAT32)) &&
            tensor.storage_type() == StorageType::DEVICE) {
            return ttnn::fill(tensor, fill_value, memory_config, optional_output_tensor);
        } else {
            return full_impl(
                queue_id,
                tensor.get_shape(),
                fill_value,
                dtype_value,
                layout_value,
                device.has_value() ? device->get_devices() : tensor.get_workers(/*blocking=*/true),
                memory_config.value_or(tensor.memory_config()),
                optional_output_tensor);
        }
    } else {
        return full_impl(
            queue_id,
            tensor.get_shape(),
            fill_value,
            dtype_value,
            layout_value,
            detail::get_workers_from_device(device),
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
    detail::OptionalAnyDevice device = std::nullopt,
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
        detail::OptionalAnyDevice device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct ZerosLike : FullLikeWith<0.0f> {};
struct OnesLike : FullLikeWith<1.0f> {};

inline constexpr ZerosLike zeros_like{};
inline constexpr OnesLike ones_like{};

struct Empty {
    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const DataType& dtype,
        const Layout& layout,
        ttnn::AnyDevice device,
        const MemoryConfig& memory_config) {
        return allocate_tensor_on_devices(shape, dtype, layout, device.get_devices(), memory_config);
    }
};

struct EmptyLike {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device_arg = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        const std::vector<IDevice*>& devices =
            device_arg.has_value() ? device_arg->get_devices() : tensor.get_workers(/*blocking=*/true);
        Layout layout_value = layout.value_or(tensor.get_layout());
        DataType dtype_value = dtype.value_or(tensor.get_dtype());
        MemoryConfig mem_cfg = memory_config.value_or(tensor.memory_config());
        return allocate_tensor_on_devices(tensor.get_shape(), dtype_value, layout_value, devices, mem_cfg);
    }
};

struct Full {
    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Shape& shape,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(
            queue_id,
            shape,
            fill_value,
            dtype,
            layout,
            detail::get_workers_from_device(device),
            memory_config,
            optional_output_tensor);
    }

    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static ttnn::Tensor invoke(
        const ttnn::Shape& shape,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_impl(
            ttnn::DefaultQueueId,
            shape,
            fill_value,
            dtype,
            layout,
            detail::get_workers_from_device(device),
            memory_config,
            optional_output_tensor);
    }
};

struct FullLike {
    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& tensor,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            queue_id, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }

    template <typename FillValueType>
        requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& tensor,
        const FillValueType fill_value,
        const std::optional<DataType>& dtype = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        detail::OptionalAnyDevice device = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt) {
        return full_like_impl(
            ttnn::DefaultQueueId, tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
    }
};

struct Arange {
    static ttnn::Tensor invoke(
        const int64_t stop,
        const DataType dtype = DataType::BFLOAT16,
        detail::OptionalAnyDevice device = std::nullopt,
        const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
        return Arange::invoke(0, stop, 1, dtype, device, memory_config);
    }

    static ttnn::Tensor invoke(
        const int64_t start,
        const int64_t stop,
        const int64_t step = 1,
        const DataType dtype = ttnn::DataType::BFLOAT16,
        detail::OptionalAnyDevice device = std::nullopt,
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

constexpr auto arange =
    ttnn::decorators::register_operation_with_auto_launch_op<"ttnn::arange", ttnn::operations::creation::Arange>();

}  // namespace ttnn

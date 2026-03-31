// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation/creation.hpp"

#include <cstdint>
#include <vector>

#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

namespace creation_detail {

template <typename T>
Tensor arange_impl(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& output_mem_config) {
    using namespace tt::tt_metal;
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

    TensorSpec spec{
        ttnn::Shape{static_cast<uint32_t>(size)}, TensorLayout{data_type, PageConfig{layout}, output_mem_config}};

    return Tensor::from_vector(
        std::move(owned_buffer), spec, device.has_value() ? std::addressof(device->get()) : nullptr);
}

template <typename T>
Tensor full_impl(
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

    Tensor host_tensor(tt::tt_metal::HostBuffer(std::move(owned_buffer)), shape, data_type, layout);

    if (optional_output_tensor.has_value()) {
        copy_to_device(host_tensor, *optional_output_tensor);
        return *optional_output_tensor;
    }
    if (device != nullptr) {
        return host_tensor.to_device(device, output_mem_config);
    }
    return host_tensor;
}

template Tensor arange_impl<::bfloat16>(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& output_mem_config);
template Tensor arange_impl<float>(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& output_mem_config);
template Tensor arange_impl<uint16_t>(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& output_mem_config);
template Tensor arange_impl<uint32_t>(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& output_mem_config);
template Tensor arange_impl<int32_t>(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const Layout layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& output_mem_config);

template Tensor full_impl<uint8_t>(
    const ttnn::Shape& shape,
    uint8_t value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_impl<uint16_t>(
    const ttnn::Shape& shape,
    uint16_t value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_impl<uint32_t>(
    const ttnn::Shape& shape,
    uint32_t value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_impl<int32_t>(
    const ttnn::Shape& shape,
    int32_t value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_impl<float>(
    const ttnn::Shape& shape,
    float value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_impl<::bfloat16>(
    const ttnn::Shape& shape,
    ::bfloat16 value,
    const Layout layout,
    MeshDevice* device,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> optional_output_tensor);

}  // namespace creation_detail

template <typename T>
Tensor full_impl(
    const ttnn::Shape& shape,
    const T fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    MeshDevice* device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    MeshDevice* device_to_use = optional_output_tensor.has_value() ? optional_output_tensor->device() : device;

    DataType dtype_value = optional_output_tensor.has_value() ? optional_output_tensor.value().dtype()
                                                              : dtype.value_or(DataType::BFLOAT16);
    auto get_default_layout = [dtype_value]() {
        return (dtype_value == DataType::BFLOAT4_B || dtype_value == DataType::BFLOAT8_B) ? ttnn::TILE_LAYOUT
                                                                                          : ttnn::ROW_MAJOR_LAYOUT;
    };

    Layout layout_value = optional_output_tensor.has_value() ? optional_output_tensor.value().layout()
                                                             : layout.value_or(get_default_layout());
    ttnn::Shape shape_value =
        optional_output_tensor.has_value() ? optional_output_tensor.value().logical_shape() : shape;
    MemoryConfig mem_cfg = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                              : memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    auto concrete_full = [&]<typename BufferType>(BufferType fill_value) {
        return creation_detail::full_impl<BufferType>(
            shape_value, fill_value, layout_value, device_to_use, mem_cfg, optional_output_tensor);
    };

    switch (dtype_value) {
        case DataType::UINT8: return concrete_full.template operator()<uint8_t>(fill_value);
        case DataType::UINT16: return concrete_full.template operator()<uint16_t>(fill_value);
        case DataType::UINT32: return concrete_full.template operator()<uint32_t>(fill_value);
        case DataType::INT32: return concrete_full.template operator()<int32_t>(fill_value);
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

template <typename T>
Tensor full_like_impl(
    const Tensor& tensor,
    const T fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device_arg,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    MeshDevice* device = device_arg.has_value() ? &device_arg->get() : nullptr;
    Layout layout_value =
        optional_output_tensor.has_value() ? optional_output_tensor.value().layout() : layout.value_or(tensor.layout());
    DataType dtype_value =
        optional_output_tensor.has_value() ? optional_output_tensor.value().dtype() : dtype.value_or(tensor.dtype());
    const bool is_tile_layout = (tensor.layout() == Layout::TILE) && (layout_value == Layout::TILE);
    if (tt::tt_metal::is_device_tensor(tensor)) {
        // requires reference tensor to be in TILE for device operation fill - this will be changed later
        if (is_tile_layout &&
            (dtype_value == DataType::BFLOAT8_B || dtype_value == DataType::BFLOAT16 ||
             dtype_value == DataType::FLOAT32) &&
            tensor.storage_type() == StorageType::DEVICE) {
            return ttnn::fill(tensor, fill_value, memory_config, optional_output_tensor);
        }
        return full_impl(
            tensor.logical_shape(),
            fill_value,
            dtype_value,
            layout_value,
            device ? device : tensor.device(),
            memory_config.value_or(tensor.memory_config()),
            optional_output_tensor);
    }
    return full_impl(
        tensor.logical_shape(),
        fill_value,
        dtype_value,
        layout_value,
        device ? device : tensor.device(),
        memory_config,
        optional_output_tensor);
}

Tensor zeros(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config) {
    return full_impl(
        shape, 0.0f, dtype, layout, device.has_value() ? &device->get() : nullptr, memory_config, std::nullopt);
}

Tensor ones(
    const ttnn::Shape& shape,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config) {
    return full_impl(
        shape, 1.0f, dtype, layout, device.has_value() ? &device->get() : nullptr, memory_config, std::nullopt);
}

template <typename FillValueType>
    requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
Tensor full(
    const ttnn::Shape& shape,
    const FillValueType fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return full_impl(
        shape,
        fill_value,
        dtype,
        layout,
        device.has_value() ? &device->get() : nullptr,
        memory_config,
        optional_output_tensor);
}

Tensor empty(
    const ttnn::Shape& shape,
    const DataType& dtype,
    const Layout& layout,
    MeshDevice* device,
    const MemoryConfig& memory_config) {
    return create_device_tensor(TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config)), device);
}

template <typename BufferType>
Tensor from_buffer(
    std::vector<BufferType>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    // This is validated from the invoker, but we need to handle it just in case that the user wants to use it
    TT_ASSERT(dtype != DataType::BFLOAT4_B && dtype != DataType::BFLOAT8_B, "Unsupported DataType!");
    TensorSpec spec(
        shape,
        TensorLayout(
            dtype, PageConfig(layout.value_or(ttnn::ROW_MAJOR_LAYOUT)), memory_config.value_or(MemoryConfig{})));
    return Tensor::from_vector<BufferType>(std::move(buffer), spec, device);
}

template <typename BufferType>
Tensor from_buffer(
    const std::vector<BufferType>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    // This is validated from the invoker, but we need to handle it just in case that the user wants to use it
    TT_ASSERT(dtype != DataType::BFLOAT4_B && dtype != DataType::BFLOAT8_B, "Unsupported DataType!");
    TensorSpec spec(
        shape,
        TensorLayout(
            dtype, PageConfig(layout.value_or(ttnn::ROW_MAJOR_LAYOUT)), memory_config.value_or(MemoryConfig{})));
    return Tensor::from_vector<BufferType>(buffer, spec, device);
}

template <typename FillValueType>
    requires std::is_same_v<FillValueType, int> or std::is_same_v<FillValueType, float>
Tensor full_like(
    const Tensor& tensor,
    const FillValueType fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return full_like_impl(tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
}

Tensor zeros_like(
    const Tensor& tensor,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return full_like_impl(tensor, 0.0f, dtype, layout, device, memory_config, optional_output_tensor);
}

Tensor ones_like(
    const Tensor& tensor,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return full_like_impl(tensor, 1.0f, dtype, layout, device, memory_config, optional_output_tensor);
}

Tensor empty_like(
    const Tensor& tensor,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config) {
    Layout layout_value = layout.value_or(tensor.layout());
    DataType dtype_value = dtype.value_or(tensor.dtype());
    MemoryConfig mem_cfg = memory_config.value_or(tensor.memory_config());
    MeshDevice* device_ptr = device.has_value() ? &device->get() : tensor.device();
    return create_device_tensor(
        TensorSpec(tensor.logical_shape(), TensorLayout(dtype_value, PageConfig(layout_value), mem_cfg)), device_ptr);
}

Tensor arange(
    const int64_t start,
    const int64_t stop,
    const int64_t step,
    const DataType dtype,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& memory_config,
    const Layout layout) {
    auto concrete_arange = [&]<typename BufferType>() {
        return creation_detail::arange_impl<BufferType>(start, stop, step, layout, device, memory_config);
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

Tensor arange(
    const int64_t stop,
    const DataType dtype,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const MemoryConfig& memory_config,
    const Layout layout) {
    return arange(0, stop, 1, dtype, device, memory_config, layout);
}

template Tensor full_impl<float>(
    const ttnn::Shape& shape,
    const float fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    MeshDevice* device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_impl<int>(
    const ttnn::Shape& shape,
    const int fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    MeshDevice* device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor);

template Tensor full_like_impl<float>(
    const Tensor& tensor,
    const float fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device_arg,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor);
template Tensor full_like_impl<int>(
    const Tensor& tensor,
    const int fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device_arg,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor);

template Tensor full<int>(
    const ttnn::Shape& shape,
    const int fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor);
template Tensor full<float>(
    const ttnn::Shape& shape,
    const float fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor);

template Tensor full_like<int>(
    const Tensor& tensor,
    const int fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor);
template Tensor full_like<float>(
    const Tensor& tensor,
    const float fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor);

template Tensor from_buffer<uint8_t>(
    std::vector<uint8_t>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<uint8_t>(
    const std::vector<uint8_t>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<uint16_t>(
    std::vector<uint16_t>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<uint16_t>(
    const std::vector<uint16_t>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<int32_t>(
    std::vector<int32_t>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<int32_t>(
    const std::vector<int32_t>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<uint32_t>(
    std::vector<uint32_t>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<uint32_t>(
    const std::vector<uint32_t>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<float>(
    std::vector<float>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<float>(
    const std::vector<float>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<::bfloat16>(
    std::vector<::bfloat16>&& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);
template Tensor from_buffer<::bfloat16>(
    const std::vector<::bfloat16>& buffer,
    const Shape& shape,
    const DataType dtype,
    MeshDevice* device,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to tilized 2d matrix layout with special block height padding
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(Tensor conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

// Converts convolution weights to depthwise layout with broadcasted weights
Tensor convert_conv_weight_tensor_to_depthwise_layout(Tensor conv_weight_tensor, uint32_t act_block_h_ntiles, DataType output_dtype);

const Shape infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume);

const Shape infer_dims_for_reshape_RM(int N, int C, int H, int W, uint32_t old_volume);

template <typename T>
static std::size_t compute_volume(const T& shape) {
    auto volume = 1;
    for (auto index = 0; index < shape.size(); index++) {
        volume *= shape[index];
    }
    return volume;
}

static std::vector<std::size_t> compute_strides(Shape shape) {
    auto num_elements = compute_volume(shape);
    std::vector<std::size_t> strides;
    for (std::int32_t index = 0; index < shape.rank(); index++) {
        num_elements /= shape[index];
        strides.push_back(num_elements);
    }
    return strides;
}

static int compute_flat_indices(vector<int> indices, vector<std::size_t> strides) {
    int flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

template <typename T>
static std::size_t compute_buffer_size(const T& shape, DataType data_type) {
    const auto volume = compute_volume(shape);
    if (data_type == DataType::BFLOAT8_B) {
        TT_ASSERT(volume % constants::TILE_HW == 0);
        const auto bfloat8_b_volume = volume / constants::TILE_HW * constants::BFLOAT8_B_TILE_HW;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat8_b_volume / sizeof(std::uint32_t);
    }
    if (data_type == DataType::BFLOAT4_B) {
        TT_ASSERT(volume % constants::TILE_HW == 0);
        const auto bfloat4_b_volume = volume / constants::TILE_HW * constants::BFLOAT4_B_TILE_HW;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat4_b_volume / sizeof(std::uint32_t);
    }
    return volume;
}

constexpr auto compute_flat_input_index = [](const auto& indices, const auto& strides) {
    uint32_t flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

bool is_arch_gs(const tt::ARCH& arch);
bool is_arch_whb0(const tt::ARCH& arch);

bool is_cpu_tensor(const Tensor& tensor);
bool is_device_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns the tensor on the given device.
Tensor get_device_tensor(const Tensor& multi_device_tensor, const Device* device);
Tensor get_device_tensor(const Tensor& multi_device_tensor, const int device_id);

// Returns true has MultiDeviceHost/MultiDevice Storage
bool is_multi_device_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns a list of per-device tensors.
std::vector<Tensor> get_tensors_from_multi_device_storage(const Tensor& multi_device_tensor);

// Given a list of per-device shards, return a multi-device tensor
Tensor create_multi_device_tensor(
    const std::vector<Tensor>& tensors, StorageType storage_type, const DistributedTensorConfig& strategy);

// Given a multi-device tensor, and a function that transforms a tensor, apply the function to all per-device
// tensors.
Tensor transform(const Tensor& tensor, std::function<Tensor(const Tensor&)> transform_func);

// Given a multi-device tensor, and a callable, apply the function to all per-device tensors.
void apply(const Tensor& tensor, std::function<void(const Tensor&)> callable);

// Given a multi-device tensor, return all the devices it is mapped to.
std::vector<Device*> get_devices(const Tensor& multi_device_tensor);

uint32_t num_buffers_in_tensor(const Tensor& tensor);

Tensor get_shard_for_device(
    const Tensor& tensor, Device* target_device, std::optional<int> buffer_index = std::nullopt);

void insert_buffer_and_shape_for_device(
    Device* target_device,
    const Tensor& shard,
    Tensor& tensor_to_modify,
    std::optional<int> buffer_index = std::nullopt);

Tensor copy_borrowed_tensor_in_async_mode(Device* worker, const Tensor& tensor);

template <typename TensorContainer>
auto get_device_tensors(Device* device, const TensorContainer& input_tensors) {
    // Could be Tensor, const Tensor, std::optional<Tensor>, or std::optional<const Tensor>
    using ValueType = typename TensorContainer::value_type;

    // We need a way to extract the underlying Tensor type (const or non-const) from ValueType
    // and to decide whether we are dealing with an optional type.
    using IsOptional = std::conditional_t<
        std::is_same_v<ValueType, std::optional<Tensor>> || std::is_same_v<ValueType, std::optional<const Tensor>>,
        std::true_type,
        std::false_type>;
    using TensorType = std::conditional_t<
        std::is_same_v<ValueType, std::optional<Tensor>> || std::is_same_v<ValueType, Tensor>,
        Tensor,
        const Tensor>;

    // Result container type adjustment based on input type
    using ResultType = std::conditional_t<IsOptional::value, std::optional<TensorType>, TensorType>;
    std::vector<ResultType> transformed_tensors;

    for (const auto& tensor : input_tensors) {
        if constexpr (IsOptional::value) {
            if (tensor.has_value()) {
                transformed_tensors.emplace_back(get_device_tensor(tensor.value(), device));
            } else {
                transformed_tensors.emplace_back(std::nullopt);
            }
        } else {
            transformed_tensors.emplace_back(get_device_tensor(tensor, device));
        }
    }
    return transformed_tensors;
}

inline bool is_tensor_on_device(const ttnn::Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

inline bool is_tensor_on_multi_device(const ttnn::Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE;
}

inline bool is_tensor_on_device_or_multidevice(const ttnn::Tensor& tensor) {
    return is_tensor_on_device(tensor) or is_tensor_on_multi_device(tensor);
}

inline bool any_tensor_on_multi_device(const std::vector<ttnn::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
        if (is_tensor_on_multi_device(tensor)) {
            return true;
        }
    }
    return false;
}

template<class T>
inline uint32_t get_batch_size(const T& shape) {
    uint32_t result = 1;
    for (auto i = 0; i < shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor);

}  // namespace tt_metal

}  // namespace tt

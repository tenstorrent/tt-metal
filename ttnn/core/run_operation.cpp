// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/old_infra_device_operation.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tracy/Tracy.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/device.hpp"

namespace tt::tt_metal::operation::detail {

distributed::MeshDevice* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors) {
    for (const auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            return input_tensor.device_storage().get_device();
        }
    }
    for (const auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor->storage_type() == StorageType::DEVICE) {
            return optional_input_tensor->device_storage().get_device();
        }
    }
    auto* device = ttnn::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to operation are on device");
    return device;
}

template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
constexpr bool is_optional_v = is_optional<T>::value;

template <class T>
Tensor* get_tensor(T& maybe_tensor) {
    Tensor* output_tensor = nullptr;
    if constexpr (is_optional_v<T>) {
        if (maybe_tensor.has_value()) {
            output_tensor = &maybe_tensor.value();
        }
    } else {
        output_tensor = &maybe_tensor;
    }
    return output_tensor;
}

}  // namespace tt::tt_metal::operation::detail

namespace tt::tt_metal::operation {

template <class OutputTensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors) {
    if constexpr (std::is_same_v<OutputTensors, Tensors>) {
        return ttnn::prim::old_infra_device_operation(
            std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        return ttnn::prim::old_infra_device_operation_with_optional_output_tensors(
            std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    }
}

template Tensors run(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors);

template OptionalTensors run(
    DeviceOperation<OptionalTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors);

std::vector<Shape> extract_padded_shapes(
    const std::vector<ttnn::TensorSpec>& tensor_specs,
    const std::function<TensorLayout(size_t idx)>& layout_provider,
    const bool use_tensor_layout_from_tensor_spec) {
    std::vector<Shape> padded_shapes;
    padded_shapes.reserve(tensor_specs.size());
    for (size_t idx = 0; idx < tensor_specs.size(); idx++) {
        const auto& tensor_spec = tensor_specs[idx];
        TensorLayout tensor_layout =
            use_tensor_layout_from_tensor_spec ? tensor_spec.tensor_layout() : layout_provider(idx);
        const auto& logical_shape = tensor_spec.logical_shape();
        padded_shapes.push_back(tensor_layout.compute_padded_shape(logical_shape));
    }
    return padded_shapes;
}

}  // namespace tt::tt_metal::operation

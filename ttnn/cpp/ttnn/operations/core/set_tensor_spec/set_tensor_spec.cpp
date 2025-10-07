// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "set_tensor_spec.hpp"

#include <tt-metalium/constants.hpp>
#include <ttnn/operations/functions.hpp>
#include "ttnn/tensor/storage.hpp"

#include <tracy/Tracy.hpp>

namespace ttnn::operations::core {

// Helper function to update device buffer metadata (page size and shard spec)
// This mirrors the logic from view.cpp
// TODO: Test this function with all kinds of memory configs and layouts
static ttnn::Tensor update_device_tensor_metadata(const ttnn::Tensor& input_tensor, const TensorSpec& new_spec) {
    auto device_storage = std::get<tt::tt_metal::DeviceStorage>(input_tensor.storage());

    // For ROW_MAJOR layout, we need to update page size
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        // Non-sharded case: just update page size
        if (input_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
            auto device_buffer = device_storage.get_buffer();
            auto page_size_bytes = new_spec.compute_page_size_bytes();
            device_buffer->set_page_size(page_size_bytes);
            return tt::tt_metal::metal_tensor::Tensor(
                std::move(device_storage), new_spec, input_tensor.tensor_topology());
        }
        // HEIGHT_SHARDED case: update shard spec and page size
        else {
            auto device_buffer = device_storage.get_buffer();
            tt::tt_metal::ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

            auto shard_spec = shard_spec_buffer.tensor_shard_spec;
            auto shard_shape = shard_spec.shape;
            const auto& new_logical_shape = new_spec.logical_shape();

            // Calculate the multiplier/divisor for updating shard height
            uint32_t mul_div;
            if (new_logical_shape[-1] == 0 || shard_shape[1] == 0) {
                mul_div = 0;
            } else {
                mul_div = new_logical_shape[-1] > shard_shape[1] ? (new_logical_shape[-1] / shard_shape[1])
                                                                 : (shard_shape[1] / new_logical_shape[-1]);
            }

            // Update shard dimensions
            shard_spec.shape[0] =
                new_logical_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div : shard_shape[0] * mul_div;
            shard_spec.shape[1] = new_logical_shape[-1];

            // Update shard spec buffer metadata
            shard_spec_buffer.page_shape = {1, new_logical_shape[-1]};
            shard_spec_buffer.tensor2d_shape_in_pages = {
                new_spec.physical_shape().height() / shard_spec_buffer.page_shape[0],
                new_spec.physical_shape().width() / shard_spec_buffer.page_shape[1]};
            shard_spec_buffer.set_shard_spec(shard_spec);
            device_buffer->set_shard_spec(shard_spec_buffer);

            // Update page size
            auto page_size_bytes = new_spec.compute_page_size_bytes();
            device_buffer->set_page_size(page_size_bytes);

            return tt::tt_metal::metal_tensor::Tensor(
                std::move(device_storage), new_spec, input_tensor.tensor_topology());
        }
    }
    // For TILE layout, just use new spec with same storage
    else {
        return tt::tt_metal::metal_tensor::Tensor(std::move(device_storage), new_spec, input_tensor.tensor_topology());
    }
}

// Compute output specs for lazy mode
SetTensorSpecOperation::spec_return_value_t SetTensorSpecOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    return new_tensor_spec;
}

void SetTensorSpecOperation::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Expected exactly one input tensor");
}

// Main invoke function - handles both eager and lazy modes
ttnn::Tensor SetTensorSpecOperation::invoke(const std::vector<Tensor>& input_tensors) const {
    // Get the materialized tensor
    const auto& tensor = input_tensors[0];

    // Update tensor based on storage type
    auto metal_output = std::visit(
        [&tensor, this](auto&& storage) -> ttnn::Tensor {
            using T = std::decay_t<decltype(storage)>;

            if constexpr (std::is_same_v<T, tt::tt_metal::DeviceStorage>) {
                return update_device_tensor_metadata(tensor, this->new_tensor_spec);
            } else if constexpr (std::is_same_v<T, tt::tt_metal::HostStorage>) {
                // For host storage, just create new tensor with same storage and new spec
                return ttnn::Tensor(tensor.storage(), this->new_tensor_spec, tensor.tensor_topology());
            } else {
                static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported storage type");
            }
        },
        tensor.storage());

    // Wrap in ttnn::Tensor and set tensor ID
    auto output = tt::tt_metal::set_tensor_id(Tensor(metal_output));
    tt::tt_metal::GraphTracker::instance().track_function_end(output);

    return output;
}

}  // namespace ttnn::operations::core

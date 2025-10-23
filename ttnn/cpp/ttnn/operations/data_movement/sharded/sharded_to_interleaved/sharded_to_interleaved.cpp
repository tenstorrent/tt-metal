// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "device/sharded_to_interleaved_op.hpp"
#include "sharded_to_interleaved.hpp"

// Forward declare redzone functions
namespace tt::tt_metal::tensor_impl::redzone {
    void verify_allocation(void* buffer_addr);
    bool is_redzone_enabled();
}

namespace ttnn::operations::data_movement {

ttnn::Tensor ShardedToInterleavedOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<DataType>& output_dtype,
    const std::optional<bool>& is_l1_aligned) {

    // RED ZONE VERIFICATION: Check input tensor before operation
    if (tt::tt_metal::tensor_impl::redzone::is_redzone_enabled()) {
        if (input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE) {
            tt::tt_metal::tensor_impl::redzone::verify_allocation(
                std::get<tt::tt_metal::DeviceStorage>(input_tensor.storage()).mesh_buffer.get()
            );
        }
    }

    if (!input_tensor.shard_spec().has_value()) {
        return input_tensor;
    }

    auto shard_spec = input_tensor.shard_spec().value();
    auto result = tt::tt_metal::operation::run(
               ShardedToInterleavedDeviceOperation{
                   .output_mem_config = memory_config,
                   .output_dtype = output_dtype.value_or(input_tensor.dtype()),
                   .is_l1_aligned = is_l1_aligned.value_or(false)},
               {input_tensor})
        .at(0);

    // RED ZONE VERIFICATION: Check output tensor after operation
    if (tt::tt_metal::tensor_impl::redzone::is_redzone_enabled()) {
        if (result.storage_type() == tt::tt_metal::StorageType::DEVICE) {
            tt::tt_metal::tensor_impl::redzone::verify_allocation(
                std::get<tt::tt_metal::DeviceStorage>(result.storage()).mesh_buffer.get()
            );
        }
    }

    return result;
}

}  // namespace ttnn::operations::data_movement

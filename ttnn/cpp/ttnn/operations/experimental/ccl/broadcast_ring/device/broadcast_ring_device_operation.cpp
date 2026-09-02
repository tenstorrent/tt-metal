// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/broadcast_ring/device/broadcast_ring_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::prim {

void BroadcastRingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "broadcast_ring input must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "broadcast_ring input must be allocated on device");
    TT_FATAL(operation_attributes.num_links > 0, "num_links must be > 0, got {}", operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.sender_ring_index < operation_attributes.ring_size,
        "sender_ring_index ({}) must be < ring_size ({})",
        operation_attributes.sender_ring_index,
        operation_attributes.ring_size);
    // v1 broadcasts one-way around the ring, so it needs the wrap link.
    TT_FATAL(
        operation_attributes.topology == tt::tt_fabric::Topology::Ring,
        "broadcast_ring v1 is one-way and requires Ring topology (wrap link)");
    // Unlike ttnn.broadcast, the orthogonal (tp) axis may be sharded: the op runs per line along the ring
    // axis, so each orthogonal row broadcasts its own data.
}

tt::tt_metal::TensorSpec BroadcastRingDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));
}

Tensor BroadcastRingDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

Tensor broadcast_ring(
    const ttnn::Tensor& input_tensor,
    uint32_t sender_ring_index,
    uint32_t cluster_axis,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    uint32_t chunk_size_tiles,
    uint32_t broadcast_offset_tiles,
    uint32_t broadcast_num_tiles) {
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(num_devices > 1, "broadcast_ring needs >1 device along cluster_axis, got {}", num_devices);

    return ttnn::device_operation::launch<BroadcastRingDeviceOperation>(
        BroadcastRingParams(
            sender_ring_index,
            cluster_axis,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            topology,
            sub_device_id,
            chunk_size_tiles,
            broadcast_offset_tiles,
            broadcast_num_tiles),
        BroadcastRingInputs{.input_tensor = input_tensor});
}

}  // namespace ttnn::prim

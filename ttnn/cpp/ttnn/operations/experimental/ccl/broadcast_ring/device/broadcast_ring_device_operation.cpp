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

    // Blocked range: num_blocks blocks of num_tiles pages, stride apart (see the params struct). Only the
    // L1 relay kernel maps linear->blocked pages; blocks must be explicit, non-overlapping and in range.
    if (operation_attributes.broadcast_num_blocks > 1) {
        TT_FATAL(operation_attributes.use_l1_relay, "broadcast_ring: blocked range requires use_l1_relay");
        TT_FATAL(
            operation_attributes.broadcast_num_tiles > 0,
            "broadcast_ring: blocked range needs an explicit broadcast_num_tiles (pages per block)");
        TT_FATAL(
            operation_attributes.broadcast_stride_pages >= operation_attributes.broadcast_num_tiles,
            "broadcast_ring: broadcast_stride_pages ({}) must be >= pages per block ({})",
            operation_attributes.broadcast_stride_pages,
            operation_attributes.broadcast_num_tiles);
        const uint32_t last_page_end =
            operation_attributes.broadcast_offset_tiles +
            (operation_attributes.broadcast_num_blocks - 1) * operation_attributes.broadcast_stride_pages +
            operation_attributes.broadcast_num_tiles;
        TT_FATAL(
            last_page_end <= input_tensor.buffer()->num_pages(),
            "broadcast_ring: blocked range ends at page {} but the shard has {} pages",
            last_page_end,
            input_tensor.buffer()->num_pages());
    }

    // Output remap: persist block b at out_offset + b*out_stride instead of the input page ids (see the
    // params struct). The output shape then differs from the input, so a caller-owned buffer is required,
    // and its page size must match the input's (the relay moves whole pages).
    if (operation_attributes.broadcast_out_stride_pages > 0) {
        TT_FATAL(operation_attributes.use_l1_relay, "broadcast_ring: output remap requires use_l1_relay");
        TT_FATAL(
            tensor_args.persistent_output_buffer.has_value(),
            "broadcast_ring: output remap requires a persistent_output_buffer (output shape != input shape)");
        const auto& out_buf = tensor_args.persistent_output_buffer.value();
        TT_FATAL(
            out_buf.buffer()->aligned_page_size() == input_tensor.buffer()->aligned_page_size(),
            "broadcast_ring: output remap buffer page size ({}) must match the input's ({})",
            out_buf.buffer()->aligned_page_size(),
            input_tensor.buffer()->aligned_page_size());
        const uint32_t block_pages = operation_attributes.broadcast_num_tiles > 0
                                         ? operation_attributes.broadcast_num_tiles
                                         : input_tensor.buffer()->num_pages();
        TT_FATAL(
            operation_attributes.broadcast_num_blocks <= 1 ||
                operation_attributes.broadcast_out_stride_pages >= block_pages,
            "broadcast_ring: broadcast_out_stride_pages ({}) must be >= pages per block ({})",
            operation_attributes.broadcast_out_stride_pages,
            block_pages);
        const uint32_t num_blocks =
            operation_attributes.broadcast_num_blocks > 1 ? operation_attributes.broadcast_num_blocks : 1;
        const uint32_t last_out_end = operation_attributes.broadcast_out_offset_pages +
                                      (num_blocks - 1) * operation_attributes.broadcast_out_stride_pages + block_pages;
        TT_FATAL(
            last_out_end <= out_buf.buffer()->num_pages(),
            "broadcast_ring: output remap ends at page {} but the output buffer has {} pages",
            last_out_end,
            out_buf.buffer()->num_pages());
    }
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
    // Trace-safe path: reuse the caller's persistent buffer (stable baked address across replays).
    if (tensor_args.persistent_output_buffer.has_value()) {
        return tensor_args.persistent_output_buffer.value();
    }
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
    uint32_t broadcast_num_tiles,
    uint32_t broadcast_stride_pages,
    uint32_t broadcast_num_blocks,
    uint32_t broadcast_out_offset_pages,
    uint32_t broadcast_out_stride_pages,
    bool use_l1_relay,
    uint32_t num_slots,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore) {
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(num_devices > 1, "broadcast_ring needs >1 device along cluster_axis, got {}", num_devices);
    // With output remap the output layout is caller-chosen (validated against the out range instead).
    if (persistent_output_buffer.has_value() && broadcast_out_stride_pages == 0) {
        TT_FATAL(
            persistent_output_buffer->logical_shape() == input_tensor.logical_shape(),
            "broadcast_ring persistent_output_buffer shape {} must match input shape {}",
            persistent_output_buffer->logical_shape(),
            input_tensor.logical_shape());
    }

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
            broadcast_num_tiles,
            broadcast_stride_pages,
            broadcast_num_blocks,
            broadcast_out_offset_pages,
            broadcast_out_stride_pages,
            use_l1_relay,
            num_slots,
            multi_device_global_semaphore),
        BroadcastRingInputs{.input_tensor = input_tensor, .persistent_output_buffer = persistent_output_buffer});
}

}  // namespace ttnn::prim

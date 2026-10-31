// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>
#include "outbound_socket_service_sync_device_operation.hpp"

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/services/d2h_socket_service.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void OutboundSocketServiceSyncOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& backing = tensor_args.backing;
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE, "outbound_socket_service_sync: input tensor must be on device");
    TT_FATAL(input.is_allocated(), "outbound_socket_service_sync: input tensor must be allocated");
    TT_FATAL(
        backing.storage_type() == StorageType::DEVICE,
        "outbound_socket_service_sync: backing tensor must be on device");
    TT_FATAL(backing.is_allocated(), "outbound_socket_service_sync: backing tensor must be allocated");
    TT_FATAL(
        input.tensor_spec() == backing.tensor_spec(),
        "outbound_socket_service_sync: input tensor spec must match the sender backing per-shard spec (the kernel "
        "copies it "
        "page-for-page)");
    TT_FATAL(args.page_size > 0 && args.num_pages > 0, "outbound_socket_service_sync: backing tensor has no pages");
    TT_FATAL(
        !args.data_ready_addrs.empty() && args.service_core_x.size() == args.data_ready_addrs.size() &&
            args.service_core_y.size() == args.data_ready_addrs.size(),
        "outbound_socket_service_sync: per-coord sender-service state is missing or inconsistent (was the service "
        "built with "
        "sender_worker_cores?)");
    TT_FATAL(
        tensor_args.extra_inputs.size() == tensor_args.extra_backings.size() &&
            args.extra_page_sizes.size() == tensor_args.extra_inputs.size() &&
            args.extra_num_pages.size() == tensor_args.extra_inputs.size(),
        "outbound_socket_service_sync: extra input / backing counts are inconsistent");
    uint32_t max_page_size = args.page_size;
    for (size_t i = 0; i < tensor_args.extra_inputs.size(); ++i) {
        const auto& extra_input = tensor_args.extra_inputs[i];
        const auto& extra_backing = tensor_args.extra_backings[i];
        TT_FATAL(
            extra_input.storage_type() == StorageType::DEVICE && extra_input.is_allocated(),
            "outbound_socket_service_sync: input tensor {} must be allocated on device",
            i + 1);
        TT_FATAL(
            extra_input.tensor_spec() == extra_backing.tensor_spec(),
            "outbound_socket_service_sync: input tensor {} spec must match the sender backing per-shard spec",
            i + 1);
        // The writer runs on the backing's mesh and reads the input by address, so an input on
        // another mesh would read unrelated memory.
        TT_FATAL(
            extra_input.device() == backing.device(),
            "outbound_socket_service_sync: input tensor {} must be on the sender backing's mesh device",
            i + 1);
        const auto* extra_buffer = extra_backing.buffer();
        TT_FATAL(
            extra_buffer != nullptr && extra_buffer->is_dram() &&
                extra_buffer->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "outbound_socket_service_sync: backing tensor {} must be DRAM interleaved",
            i + 1);
        TT_FATAL(
            args.extra_page_sizes[i] > 0 && args.extra_num_pages[i] > 0,
            "outbound_socket_service_sync: backing tensor {} has no pages",
            i + 1);
        max_page_size = std::max(max_page_size, args.extra_page_sizes[i]);
    }
    if (args.metadata_only) {
        TT_FATAL(args.metadata_size_bytes > 0, "outbound_socket_service_sync: metadata_only requires metadata");
    }
    if (args.metadata_size_bytes > 0) {
        TT_FATAL(
            args.metadata_size_bytes % 4 == 0,
            "outbound_socket_service_sync: metadata_size_bytes must be a multiple of 4 (uint32-aligned), got {}",
            args.metadata_size_bytes);
        TT_FATAL(
            args.metadata_size_bytes <= max_page_size,
            "outbound_socket_service_sync: metadata_size_bytes ({}) exceeds the largest per-shard page_size ({}); the "
            "metadata blob is staged through the page-sized scratch CB and would overrun it",
            args.metadata_size_bytes,
            max_page_size);
        TT_FATAL(
            tensor_args.metadata.has_value() && tensor_args.metadata->is_allocated(),
            "outbound_socket_service_sync: metadata_size_bytes > 0 requires an allocated metadata tensor");
        TT_FATAL(
            args.metadata_addrs.size() == args.data_ready_addrs.size(),
            "outbound_socket_service_sync: per-coord metadata addresses are inconsistent");
    }
}

OutboundSocketServiceSyncOperation::spec_return_value_t OutboundSocketServiceSyncOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // The op fills the service-owned backing in place; its spec is the output spec.
    return tensor_args.backing.tensor_spec();
}

OutboundSocketServiceSyncOperation::topology_return_value_t OutboundSocketServiceSyncOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Empty keeps the default. With extra inputs the default would merge their placements into
    // the output, which aliases the backing, so pin it to the backing's own topology.
    if (tensor_args.extra_inputs.empty()) {
        return {};
    }
    return {tensor_args.backing.tensor_topology()};
}

OutboundSocketServiceSyncOperation::tensor_return_value_t OutboundSocketServiceSyncOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // In-place: hand back the existing sender backing tensor (no new allocation). The
    // writer kernel writes into its buffer via a BufferBinding.
    return tensor_args.backing;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Shared launch body, templated on the service type (mirrors inbound_socket_service_sync).
// D2DStreamServiceSender and D2HStreamService expose the same sender-side getters, so one
// path drives both. Modes:
//   * input + metadata : copy input into the service backing tensor and forward metadata.
//   * input only       : copy input into the backing tensor (no metadata).
//   * metadata only    : no backing/DRAM payload (D2H global_spec=None); forward the record.
template <typename ServiceT>
ttnn::Tensor outbound_socket_service_sync_impl(
    const ServiceT& service,
    const std::optional<ttnn::Tensor>& input,
    const std::optional<ttnn::Tensor>& metadata,
    const std::vector<ttnn::Tensor>& extra_inputs = {}) {
    using OperationType = ttnn::experimental::prim::OutboundSocketServiceSyncOperation;
    TT_FATAL(
        input.has_value() || metadata.has_value(),
        "outbound_socket_service_sync: pass an input (backing) tensor, metadata, or both");
    const bool metadata_only = !input.has_value();
    // Tensor mode copies input -> the service backing tensor (page geometry from the
    // backing). Metadata-only has no backing, so the record supplies geometry and stands in
    // for the (kernel-unused) input/backing slots.
    const ttnn::Tensor& backing = metadata_only ? *metadata : service.get_backing_tensor();
    const ttnn::Tensor& input_t = metadata_only ? *metadata : *input;
    auto* mesh_device = backing.device();
    const auto& mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "outbound_socket_service_sync: expects a 2D mesh, got {} dims", mesh_shape.dims());
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];
    OperationType::operation_attributes_t attrs;
    attrs.page_size = static_cast<uint32_t>(backing.buffer()->page_size());
    attrs.num_pages = static_cast<uint32_t>(backing.buffer()->num_pages());
    std::vector<ttnn::Tensor> extra_backings;
    if constexpr (std::is_same_v<ServiceT, ttnn::D2DStreamServiceSender>) {
        const auto& backings = service.get_backing_tensors();
        TT_FATAL(
            extra_inputs.size() + 1 == backings.size(),
            "outbound_socket_service_sync: the sender streams {} tensors but {} inputs were passed",
            backings.size(),
            extra_inputs.size() + 1);
        extra_backings.assign(backings.begin() + 1, backings.end());
        for (const auto& extra_backing : extra_backings) {
            attrs.extra_page_sizes.push_back(static_cast<uint32_t>(extra_backing.buffer()->page_size()));
            attrs.extra_num_pages.push_back(static_cast<uint32_t>(extra_backing.buffer()->num_pages()));
        }
    } else {
        TT_FATAL(extra_inputs.empty(), "outbound_socket_service_sync: extra inputs are D2D-only");
    }
    attrs.scratch_cb_index = 0;
    attrs.metadata_size_bytes =
        metadata.has_value() ? static_cast<uint32_t>(metadata->logical_shape()[-1] * sizeof(uint32_t)) : 0;
    attrs.metadata_only = metadata_only;
    attrs.worker_cores = service.get_worker_cores();
    attrs.mesh_num_cols = num_cols;
    const bool has_metadata = attrs.metadata_size_bytes > 0;
    if constexpr (std::is_same_v<ServiceT, tt::tt_metal::D2HStreamService>) {
        if (has_metadata) {
            attrs.metadata_ring_slots = service.get_metadata_ring_slots();
            attrs.metadata_slot_stride = service.get_metadata_slot_stride();
            attrs.metadata_data_offset = service.get_metadata_data_offset();
        }
    }
    attrs.data_ready_addrs.reserve(num_rows * num_cols);
    attrs.service_core_x.reserve(num_rows * num_cols);
    attrs.service_core_y.reserve(num_rows * num_cols);
    if (has_metadata) {
        attrs.metadata_addrs.reserve(num_rows * num_cols);
    }
    for (uint32_t row = 0; row < num_rows; ++row) {
        for (uint32_t col = 0; col < num_cols; ++col) {
            const ttnn::MeshCoordinate coord(row, col);
            attrs.data_ready_addrs.push_back(static_cast<uint32_t>(service.get_data_ready_counter_addr(coord)));
            const CoreCoord service_logical = service.get_service_core(coord);
            attrs.service_core_x.push_back(static_cast<uint32_t>(service_logical.x));
            attrs.service_core_y.push_back(static_cast<uint32_t>(service_logical.y));
            if (has_metadata) {
                attrs.metadata_addrs.push_back(static_cast<uint32_t>(service.get_metadata_addr(coord)));
            }
            if constexpr (std::is_same_v<ServiceT, tt::tt_metal::D2HStreamService>) {
                if (attrs.metadata_ring_slots > 1) {
                    attrs.termination_addrs.push_back(static_cast<uint32_t>(service.get_termination_addr(coord)));
                }
            }
        }
    }
    return ttnn::device_operation::launch<OperationType>(
        attrs, OperationType::tensor_args_t{input_t, backing, metadata, extra_inputs, std::move(extra_backings)});
}

ttnn::Tensor outbound_socket_service_sync(
    const ttnn::D2DStreamServiceSender& service,
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& metadata) {
    return outbound_socket_service_sync_impl(service, input, metadata);
}
std::vector<ttnn::Tensor> outbound_socket_service_sync(
    const ttnn::D2DStreamServiceSender& service,
    const std::vector<ttnn::Tensor>& inputs,
    const std::optional<ttnn::Tensor>& metadata) {
    TT_FATAL(!inputs.empty(), "outbound_socket_service_sync: inputs must not be empty");
    outbound_socket_service_sync_impl(
        service, inputs.front(), metadata, std::vector<ttnn::Tensor>(inputs.begin() + 1, inputs.end()));
    return service.get_backing_tensors();
}
ttnn::Tensor outbound_socket_service_sync(
    const tt::tt_metal::D2HStreamService& service,
    const std::optional<ttnn::Tensor>& input,
    const std::optional<ttnn::Tensor>& metadata) {
    return outbound_socket_service_sync_impl(service, input, metadata);
}

}  // namespace ttnn::prim

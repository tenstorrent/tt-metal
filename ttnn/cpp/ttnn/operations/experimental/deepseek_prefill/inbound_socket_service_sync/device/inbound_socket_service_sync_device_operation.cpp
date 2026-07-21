// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "inbound_socket_service_sync_device_operation.hpp"

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

InboundSocketServiceSyncOperation::program_factory_t InboundSocketServiceSyncOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return InboundSocketServiceSyncProgramFactory{};
}

void InboundSocketServiceSyncOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& backing = tensor_args.backing;
    TT_FATAL(
        backing.storage_type() == StorageType::DEVICE, "inbound_socket_service_sync: backing tensor must be on device");
    TT_FATAL(backing.buffer() != nullptr, "inbound_socket_service_sync: backing tensor must be allocated");
    TT_FATAL(args.page_size > 0 && args.num_pages > 0, "inbound_socket_service_sync: backing tensor has no pages");
    TT_FATAL(
        !args.consumed_addrs.empty() && args.service_core_x.size() == args.consumed_addrs.size() &&
            args.service_core_y.size() == args.consumed_addrs.size(),
        "inbound_socket_service_sync: per-coord service state is missing or inconsistent (was the service built with "
        "worker_cores?)");
    if (args.metadata_size_bytes > 0) {
        TT_FATAL(
            args.metadata_size_bytes % 4 == 0,
            "inbound_socket_service_sync: metadata_size_bytes must be a multiple of 4 (uint32-aligned), got {}",
            args.metadata_size_bytes);
    }
}

InboundSocketServiceSyncOperation::spec_return_value_t InboundSocketServiceSyncOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    std::vector<tt::tt_metal::TensorSpec> specs;
    // tokens: identical per-shard spec to the backing tensor.
    specs.push_back(tensor_args.backing.tensor_spec());
    // metadata: [1,1,1, metadata_size_bytes/4] uint32 ROW_MAJOR DRAM, replicated.
    if (args.metadata_size_bytes > 0) {
        const ttnn::Shape meta_shape({1u, 1u, 1u, args.metadata_size_bytes / 4u});
        specs.push_back(tt::tt_metal::TensorSpec(
            meta_shape,
            TensorLayout(
                DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM})));
    }
    return specs;
}

InboundSocketServiceSyncOperation::tensor_return_value_t InboundSocketServiceSyncOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.backing.device();
    std::vector<Tensor> outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(create_device_tensor(spec, device));
    }
    return outputs;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

namespace {

// Shared launch body. Snapshots the per-coord service state out of `service`,
// then runs the cached device operation. Templated on the service type so the
// same op drains either an H2DStreamService (host->device path) or a
// D2DStreamServiceReceiver (device->device path): both expose the identical
// receiver-side getters (only the address return width differs, absorbed by the
// static_casts below).
template <typename ServiceT>
std::vector<ttnn::Tensor> inbound_socket_service_sync_impl(const ServiceT& service, uint32_t metadata_size_bytes) {
    using OperationType = ttnn::experimental::prim::InboundSocketServiceSyncOperation;

    const auto& backing = service.get_backing_tensor();
    auto* mesh_device = backing.device();
    const auto& mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "inbound_socket_service_sync: expects a 2D mesh, got {} dims", mesh_shape.dims());
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    OperationType::operation_attributes_t attrs;
    attrs.data_ready_sem_addr = static_cast<uint32_t>(service.get_data_ready_sem_addr());
    attrs.page_size = static_cast<uint32_t>(backing.buffer()->page_size());
    attrs.num_pages = static_cast<uint32_t>(backing.buffer()->num_pages());
    attrs.scratch_cb_index = 0;
    attrs.metadata_size_bytes = metadata_size_bytes;
    attrs.metadata_l1_addr = metadata_size_bytes > 0 ? static_cast<uint32_t>(service.get_metadata_addr()) : 0;
    attrs.worker_cores = service.get_worker_cores();
    attrs.mesh_num_cols = num_cols;

    attrs.consumed_addrs.reserve(num_rows * num_cols);
    attrs.service_core_x.reserve(num_rows * num_cols);
    attrs.service_core_y.reserve(num_rows * num_cols);
    for (uint32_t row = 0; row < num_rows; ++row) {
        for (uint32_t col = 0; col < num_cols; ++col) {
            const ttnn::MeshCoordinate coord(row, col);
            attrs.consumed_addrs.push_back(static_cast<uint32_t>(service.get_consumed_counter_addr(coord)));
            const CoreCoord service_logical = service.get_service_core(coord);
            attrs.service_core_x.push_back(static_cast<uint32_t>(service_logical.x));
            attrs.service_core_y.push_back(static_cast<uint32_t>(service_logical.y));
        }
    }

    return ttnn::device_operation::launch<OperationType>(attrs, OperationType::tensor_args_t{backing});
}

}  // namespace

std::vector<ttnn::Tensor> inbound_socket_service_sync(
    const tt::tt_metal::H2DStreamService& service, uint32_t metadata_size_bytes) {
    return inbound_socket_service_sync_impl(service, metadata_size_bytes);
}

std::vector<ttnn::Tensor> inbound_socket_service_sync(
    const ttnn::D2DStreamServiceReceiver& service, uint32_t metadata_size_bytes) {
    return inbound_socket_service_sync_impl(service, metadata_size_bytes);
}

}  // namespace ttnn::prim

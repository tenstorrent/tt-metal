// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "d2d_socket_sync_device_operation.hpp"

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void D2DSocketSyncOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& backing = tensor_args.backing;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "d2d_socket_sync: input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "d2d_socket_sync: input tensor must be allocated");
    TT_FATAL(backing.storage_type() == StorageType::DEVICE, "d2d_socket_sync: backing tensor must be on device");
    TT_FATAL(backing.buffer() != nullptr, "d2d_socket_sync: backing tensor must be allocated");
    TT_FATAL(
        input.tensor_spec() == backing.tensor_spec(),
        "d2d_socket_sync: input tensor spec must match the sender backing per-shard spec (the kernel copies it "
        "page-for-page)");
    TT_FATAL(args.page_size > 0 && args.num_pages > 0, "d2d_socket_sync: backing tensor has no pages");
    TT_FATAL(
        !args.data_ready_addrs.empty() && args.service_core_x.size() == args.data_ready_addrs.size() &&
            args.service_core_y.size() == args.data_ready_addrs.size(),
        "d2d_socket_sync: per-coord sender-service state is missing or inconsistent (was the service built with "
        "sender_worker_cores?)");
    if (args.metadata_size_bytes > 0) {
        TT_FATAL(
            args.metadata_size_bytes % 4 == 0,
            "d2d_socket_sync: metadata_size_bytes must be a multiple of 4 (uint32-aligned), got {}",
            args.metadata_size_bytes);
        TT_FATAL(
            tensor_args.metadata.has_value() && tensor_args.metadata->buffer() != nullptr,
            "d2d_socket_sync: metadata_size_bytes > 0 requires an allocated metadata tensor");
        TT_FATAL(
            args.metadata_addrs.size() == args.data_ready_addrs.size(),
            "d2d_socket_sync: per-coord metadata addresses are inconsistent");
    }
}

D2DSocketSyncOperation::spec_return_value_t D2DSocketSyncOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // The op fills the service-owned backing in place; its spec is the output spec.
    return tensor_args.backing.tensor_spec();
}

D2DSocketSyncOperation::tensor_return_value_t D2DSocketSyncOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // In-place: hand back the existing sender backing tensor (no new allocation). The
    // writer kernel writes into its buffer via a BufferBinding.
    return tensor_args.backing;
}

tt::stl::hash::hash_t D2DSocketSyncOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Stable across calls for a given (service, config): the changing input address is
    // a BufferBinding (patched on cache hit), NOT part of the hash.
    return operation::hash_operation<D2DSocketSyncOperation>(
        args.page_size,
        args.num_pages,
        args.scratch_cb_index,
        args.metadata_size_bytes,
        args.worker_cores,
        args.mesh_num_cols,
        args.data_ready_addrs,
        args.service_core_x,
        args.service_core_y,
        args.metadata_addrs,
        tensor_args.input.dtype());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor d2d_socket_sync(
    const tt::tt_metal::D2DStreamServiceSender& service,
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& metadata) {
    using OperationType = ttnn::experimental::prim::D2DSocketSyncOperation;

    const auto& backing = service.get_backing_tensor();
    auto* mesh_device = backing.device();
    const auto& mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "d2d_socket_sync: expects a 2D mesh, got {} dims", mesh_shape.dims());
    const uint32_t num_rows = mesh_shape[0];
    const uint32_t num_cols = mesh_shape[1];

    OperationType::operation_attributes_t attrs;
    attrs.page_size = static_cast<uint32_t>(backing.buffer()->page_size());
    attrs.num_pages = static_cast<uint32_t>(backing.buffer()->num_pages());
    attrs.scratch_cb_index = 0;
    attrs.metadata_size_bytes =
        metadata.has_value() ? static_cast<uint32_t>(metadata->logical_shape()[-1] * sizeof(uint32_t)) : 0;
    attrs.worker_cores = service.get_worker_cores();
    attrs.mesh_num_cols = num_cols;

    const bool has_metadata = attrs.metadata_size_bytes > 0;
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
        }
    }

    return ttnn::device_operation::launch<OperationType>(attrs, OperationType::tensor_args_t{input, backing, metadata});
}

}  // namespace ttnn::prim

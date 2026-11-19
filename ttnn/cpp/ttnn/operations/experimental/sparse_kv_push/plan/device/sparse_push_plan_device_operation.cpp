// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_plan_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using ::sparse_push_plan::kChunkRows;

// What the program hash leaves out, checked on every dispatch: buffers (their addresses are patched in) and the devices
// they live on.
void validate_runtime(const SparsePushPlanParams& attrs, const SparsePushPlanInputs& args) {
    TT_FATAL(
        !attrs.chunk_start.has_value() || *attrs.chunk_start < attrs.max_rows,
        "sparse push plan chunk_start {} must be below max_rows {}",
        attrs.chunk_start.value_or(0),
        attrs.max_rows);
    const auto& idx = args.indices;
    TT_FATAL(
        idx.storage_type() == StorageType::DEVICE && idx.buffer() != nullptr &&
            idx.buffer()->buffer_type() == BufferType::DRAM,
        "indices must be allocated in DRAM");
    if (!args.chunk_start_tensor.has_value()) {
        return;
    }
    const auto& start = *args.chunk_start_tensor;
    TT_FATAL(
        start.storage_type() == StorageType::DEVICE && start.buffer() != nullptr && start.device() == idx.device(),
        "chunk_start must be allocated on the indices' device");
    // A chunk start sharing pages with the indices would read a top-k value as the start. Both are interleaved over
    // the DRAM banks: they overlap exactly when their per-bank ranges do.
    const auto* idx_buf = idx.buffer();
    const auto* start_buf = start.buffer();
    TT_FATAL(
        idx_buf->address() + idx_buf->aligned_size_per_bank() <= start_buf->address() ||
            start_buf->address() + start_buf->aligned_size_per_bank() <= idx_buf->address(),
        "chunk_start must not overlap the indices");
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

void SparsePushPlanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_runtime(attrs, args);
    const auto& idx = args.indices;
    TT_FATAL(idx.dtype() == DataType::UINT32 && idx.layout() == Layout::ROW_MAJOR, "indices must be UINT32 ROW_MAJOR");
    TT_FATAL(
        idx.padded_shape().rank() == 4 && idx.padded_shape()[0] == 1 && idx.padded_shape()[1] == 1,
        "indices must be [1, 1, rows, topk], got {}",
        idx.padded_shape());
    const uint32_t topk = idx.padded_shape()[3];
    TT_FATAL(topk % kPlanIdxAlign == 0, "sparse push plan needs top-k {} to be a multiple of {}", topk, kPlanIdxAlign);
    // Every chip of the mesh owns a share of the cache.
    const uint32_t num_devices = idx.device()->shape().mesh_size();
    TT_FATAL(
        num_devices <= ::sparse_push::plan_table::kMaxDevices,
        "the plan table holds at most {} owners, got a {}-chip mesh",
        ::sparse_push::plan_table::kMaxDevices,
        num_devices);
    validate_stripe_chunk(attrs.stripe_chunk, attrs.max_rows);
    TT_FATAL(
        attrs.max_rows % kChunkRows == 0,
        "sparse push plan max_rows {} must be a multiple of {} (its reduce chunk)",
        attrs.max_rows,
        kChunkRows);
    const uint32_t chunks = attrs.max_rows / kChunkRows, cores = plan_cores(idx.device());
    TT_FATAL(
        tt::div_up(chunks, cores) <= ::sparse_push_plan::kMaxChunksPerCore,
        "sparse push plan: {} chunks over {} cores exceed {} reduce chunks per core",
        chunks,
        cores,
        ::sparse_push_plan::kMaxChunksPerCore);
    packet_shape(idx.device(), attrs.page_bytes);  // checks page_bytes against the fabric payload
    TT_FATAL(
        !(attrs.chunk_start.has_value() && args.chunk_start_tensor.has_value()),
        "sparse push plan takes chunk_start or chunk_start_tensor, not both");
    if (attrs.dense_below > 0) {
        TT_FATAL(
            attrs.chunk_start.has_value() || args.chunk_start_tensor.has_value(),
            "sparse push plan dense_below needs chunk_start or chunk_start_tensor");
        TT_FATAL(
            uint64_t{dense_slot_rows(attrs, num_devices)} * num_devices <= attrs.max_rows,
            "sparse push plan: dense_below {} needs {} rows per chip, more than the cache's {}",
            attrs.dense_below,
            dense_slot_rows(attrs, num_devices),
            attrs.max_rows / num_devices);
    }
    if (args.chunk_start_tensor.has_value()) {
        const auto& start = *args.chunk_start_tensor;
        TT_FATAL(
            start.dtype() == DataType::UINT32 && start.logical_volume() == 1 &&
                start.memory_config().buffer_type() == BufferType::DRAM,
            "chunk_start must be a 1-element UINT32 DRAM tensor");
    }
}

void SparsePushPlanDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_runtime(attrs, args);
}

ttsl::hash::hash_t SparsePushPlanDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // A host chunk start is a runtime arg: only whether there is one, and in which form, shapes the program.
    return tt::tt_metal::operation::hash_operation<SparsePushPlanDeviceOperation>(
        attrs.stripe_chunk, attrs.max_rows, attrs.page_bytes, attrs.chunk_start.has_value(), attrs.dense_below, args);
}

SparsePushPlanDeviceOperation::spec_return_value_t SparsePushPlanDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto layout = PlanTableLayout::make(attrs.max_rows, attrs.stripe_chunk);
    const tt::tt_metal::TensorLayout rm(
        DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM});
    return {
        tt::tt_metal::TensorSpec(Shape({1, 1, layout.pages, PlanTableLayout::kPageWords}), rm),
        tt::tt_metal::TensorSpec(args.indices.logical_shape(), rm)};
}

SparsePushPlanDeviceOperation::topology_return_value_t SparsePushPlanDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& args) {
    // Each chip plans its own top-k, so the tables differ: sharded on their rows over every chip in row-major order,
    // the order high_bw_all_gather's full-mesh ring gathers them in. The compact indices are per chip, like the
    // indices they rewrite.
    return {rows_over_every_chip(args.indices.device()->shape()), args.indices.tensor_topology()};
}

SparsePushPlanDeviceOperation::tensor_return_value_t SparsePushPlanDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto specs = compute_output_specs(attrs, args);
    return {
        create_device_tensor(specs[0], args.indices.device()), create_device_tensor(specs[1], args.indices.device())};
}

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    std::optional<uint32_t> chunk_start,
    const std::optional<Tensor>& chunk_start_tensor,
    uint32_t dense_below) {
    using Op = ttnn::operations::experimental::sparse_kv_push::SparsePushPlanDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .stripe_chunk = stripe_chunk,
            .max_rows = max_rows,
            .page_bytes = page_bytes,
            .chunk_start = chunk_start,
            .dense_below = dense_below},
        Op::tensor_args_t{.indices = indices, .chunk_start_tensor = chunk_start_tensor});
}

}  // namespace ttnn::prim

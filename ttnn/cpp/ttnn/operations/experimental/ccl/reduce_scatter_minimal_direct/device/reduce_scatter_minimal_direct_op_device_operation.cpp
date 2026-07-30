// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_direct_op_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/host_api.hpp>

#include <algorithm>

namespace ttnn::experimental::prim {

using namespace ::ttnn::ccl;

namespace {

// Staging placement, in preference order.
//
// SHARDED (the fast path): one L1 shard per worker core, aliased directly into that core's reduce CB, so
// a sender's fabric packet lands in the exact CB slot the reducer unpacks from and the receive side never
// reads staging at all. That readback is pure post-gate latency -- it happens strictly after the last
// arrival, and even out of L1 it is N-1 NoC round trips plus a barrier on the critical path. Requires the
// whole per-core shard (both parity halves, every source, every chunk this core owns) to fit the
// per-core budget below.
//
// INTERLEAVED L1 / DRAM (fallbacks): the original layout, read back by the reader over the NoC. Anything
// whose shard is too big for one core still runs, just without the aliasing win.
//
// Both the factory and the persistent-buffer helper get this from compute_output_specs, so there is a
// single decision point; the factory recognises the fast path by the memory layout being sharded.
constexpr uint64_t k_l1_staging_budget_bytes = 4ull << 20;  // interleaved-L1 fallback, total across banks
constexpr uint64_t k_l1_shard_budget_bytes = 640ull << 10;  // aliased-CB path, PER worker core

}  // namespace

// Chunk-paged staging spec, doubled: the first half serves even invocations, the second odd ones (see the
// reader kernel's parity note).
//
// Interleaved layout: 2 * num_devices * chunks_per_slice pages of page_bytes, page
// (half + s * chunks_per_slice + chunk) holding source device s's contribution to our slice.
//
// Sharded layout: the same content re-grouped so it can double as the reduce CB, which forces
// chunk-major ordering (the CB read pointer advances one N-block group per chunk) and per-core scoping
// (a core only stages the chunks it owns). Row (half + k * num_devices + b) of a core's shard holds
// block b of that core's k-th chunk: b == 0 is the core's own contribution, written locally by the
// reader, and b == 1..N-1 are the remote sources in ascending device order skipping ourselves. Rows are
// sized by chunks_per_slice rather than the actual per-core chunk count so the spec stays independent of
// how the chunks happen to be partitioned across workers.
static ttnn::TensorSpec reduce_scatter_direct_staging_spec(
    const ReduceScatterMinimalDirectParams& args, const ttnn::Tensor& input_tensor, bool fp32_dest_acc_en) {
    const uint32_t num_devices = args.num_devices;
    // A slice is chunked as one flat page run (chunks_per_slice), whatever dim it was cut on -- the
    // reader walks it linearly, so staging never needs a per-batch/channel axis.
    const auto geom = reduce_scatter_direct_geometry(args, input_tensor, fp32_dest_acc_en);
    const uint32_t total_chunks = num_devices * geom.chunks_per_slice;

    const auto row_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    const uint32_t shard_rows = 2 * geom.chunks_per_slice * num_devices;
    const uint64_t shard_bytes = uint64_t{shard_rows} * geom.page_bytes;

    if (shard_bytes <= k_l1_shard_budget_bytes) {
        // Sharded over the WHOLE compute grid, deliberately, rather than over the worker cores: this spec
        // is also what reduce_scatter_minimal_direct_create_persistent_buffers allocates from, and that
        // helper cannot know the resolved num_links / subdevice / sub_core_grid. Sizing the shard by
        // chunks_per_slice (the most chunks any single worker could own) and covering every core the
        // factory could possibly pick makes the spec depend only on the input, so a persistent buffer can
        // never disagree with the program about the shard geometry. Workers are always a subset of this
        // grid, and each of them finds its shard at the same address.
        const auto grid = input_tensor.device()->compute_with_storage_grid_size();
        const CoreRangeSet shard_grid(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        const uint32_t num_shards = grid.x * grid.y;
        // Height-sharded: every shard sits at the SAME L1 address on every core of every device, which is
        // what lets a sender address the mirror core's reduce CB with nothing but its own staging address.
        tt::tt_metal::ShardSpec shard_spec(
            shard_grid, {shard_rows, geom.page_bytes}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        return ttnn::TensorSpec(
            ttnn::Shape({num_shards * shard_rows, geom.page_bytes}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT8,
                row_config,
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec)));
    }

    const uint64_t total_bytes = 2ull * total_chunks * geom.page_bytes;
    const auto buffer_type =
        total_bytes <= k_l1_staging_budget_bytes ? tt::tt_metal::BufferType::L1 : tt::tt_metal::BufferType::DRAM;

    // Opaque byte-staging, same layout contract as the ring op's intermediate: row-major UINT8, page (row)
    // = one chunk (page_bytes, DRAM-aligned). Interleaved so chunks spread across banks. The mesh
    // allocator is lockstep, so the buffer lands at the same address on every device -- required, since a
    // sender computes the destination address from its own accessor.
    return ttnn::TensorSpec(
        ttnn::Shape({2 * total_chunks, geom.page_bytes}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT8,
            row_config,
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type)));
}

void ReduceScatterMinimalDirectDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

void ReduceScatterMinimalDirectDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in a buffer on device!");

    const int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    TT_FATAL(args.dim >= 0 && args.dim < rank, "Resolved scatter dim {} out of range for {}D input", args.dim, rank);
    TT_FATAL(args.num_devices > 1, "reduce_scatter collective requires num_devices > 1, got {}", args.num_devices);

    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(args.fabric_config);
    TT_FATAL(!fabric_is_2d, "reduce_scatter_minimal_direct supports Fabric_1D ring only, not Fabric_2D");
    TT_FATAL(
        input_tensor.logical_shape()[args.dim] % args.num_devices == 0,
        "scatter dim {} (size {}) must be divisible by num_devices {}",
        args.dim,
        input_tensor.logical_shape()[args.dim],
        args.num_devices);

    auto specs = compute_output_specs(args, tensor_args);
    if (tensor_args.persistent_output_tensor.has_value()) {
        const auto& out = tensor_args.persistent_output_tensor.value();
        TT_FATAL(out.storage_type() == StorageType::DEVICE, "Persistent output tensor must be on device!");
        TT_FATAL(
            out.dtype() == input_tensor.dtype(),
            "Output dtype {} must match input {}",
            out.dtype(),
            input_tensor.dtype());
        TT_FATAL(
            out.logical_shape() == specs.at(0).logical_shape(),
            "Persistent output shape {} must be {}",
            out.logical_shape(),
            specs.at(0).logical_shape());
    }
    if (tensor_args.persistent_staging_tensor.has_value()) {
        const auto& staging = tensor_args.persistent_staging_tensor.value();
        TT_FATAL(staging.storage_type() == StorageType::DEVICE, "Persistent staging tensor must be on device!");
        TT_FATAL(
            staging.logical_shape() == specs.at(1).logical_shape() &&
                staging.memory_config() == specs.at(1).memory_config(),
            "Persistent staging tensor must come from reduce_scatter_minimal_direct_create_persistent_buffers: "
            "expected shape {} in {}, got shape {} in {}",
            specs.at(1).logical_shape(),
            specs.at(1).memory_config(),
            staging.logical_shape(),
            staging.memory_config());
    }
}

ReduceScatterMinimalDirectDeviceOperation::spec_return_value_t
ReduceScatterMinimalDirectDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto page_config = input_tensor.tensor_spec().page_config();
    const auto dtype = input_tensor.dtype();

    // [0] output slice: input shape with the scatter dim reduced by num_devices.
    auto out_shape = input_tensor.logical_shape();
    out_shape[args.dim] /= args.num_devices;
    TensorSpec output_spec(out_shape, tt::tt_metal::TensorLayout(dtype, page_config, args.output_mem_config));

    // [1] staging for the incoming contributions (see reduce_scatter_direct_staging_spec).
    const bool fp32_dest_acc_en = (dtype == DataType::FLOAT32);
    auto staging_spec = reduce_scatter_direct_staging_spec(args, input_tensor, fp32_dest_acc_en);

    return {output_spec, staging_spec};
}

ReduceScatterMinimalDirectDeviceOperation::tensor_return_value_t
ReduceScatterMinimalDirectDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input_tensor.device();

    Tensor output = tensor_args.persistent_output_tensor.has_value() ? tensor_args.persistent_output_tensor.value()
                                                                     : create_device_tensor(specs.at(0), device);
    Tensor staging = tensor_args.persistent_staging_tensor.has_value() ? tensor_args.persistent_staging_tensor.value()
                                                                       : create_device_tensor(specs.at(1), device);
    return {std::move(output), std::move(staging)};
}

ReduceScatterMinimalDirectDeviceOperation::program_factory_t
ReduceScatterMinimalDirectDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return program_factory_t{ReduceScatterMinimalDirectMeshWorkloadFactory{}};
}

// Build the operation attributes + inputs by querying the machine/fabric setup, mirroring the unicast op.
static std::tuple<ReduceScatterMinimalDirectParams, ReduceScatterMinimalDirectInputs> build_operation_args(
    const Tensor& input_tensor,
    int32_t dim,
    const MemoryConfig& output_mem_config,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    const std::optional<Tensor>& persistent_output_tensor,
    const std::optional<Tensor>& persistent_staging_tensor,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Input tensor must be on a mesh device for reduce_scatter_minimal_direct");

    const auto mesh_shape = mesh_device->shape();
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    std::array<tt::tt_fabric::Topology, 2> axis_topology{
        tt::tt_fabric::Topology::Linear, tt::tt_fabric::Topology::Linear};
    std::array<uint32_t, 2> axis_num_devices{1u, 1u};
    std::array<uint32_t, 2> axis_num_links{0u, 0u};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        const bool is_axis_active = mesh_shape[axis] > 1 && cluster_axis.value_or(axis) == axis;
        if (!is_axis_active) {
            continue;
        }
        axis_topology[axis] = ::ttnn::ccl::get_axis_topology(input_tensor, fabric_config, axis);
        axis_num_devices[axis] = ::ttnn::ccl::get_topological_dimension(input_tensor, axis);
        axis_num_links[axis] = ttnn::operations::ccl::common::get_num_links(*mesh_device, axis);
    }
    const uint32_t num_devices = axis_num_devices[0] * axis_num_devices[1];
    const size_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    uint32_t active_axis = 0;
    for (uint32_t a = 0; a < 2; ++a) {
        if (axis_num_devices[a] > 1) {
            active_axis = a;
        }
    }
    const uint32_t available_links = std::max(1u, axis_num_links[cluster_axis.value_or(active_axis)]);
    const uint32_t resolved_num_links = num_links.value_or(available_links);
    TT_FATAL(
        resolved_num_links >= 1 && resolved_num_links <= available_links,
        "reduce_scatter_minimal_direct num_links {} must be in [1, {}] (links available on axis {})",
        resolved_num_links,
        available_links,
        cluster_axis.value_or(active_axis));

    const uint32_t rank = input_tensor.logical_shape().rank();
    const int32_t scatter_dim = (dim < 0) ? static_cast<int32_t>(rank) + dim : dim;

    return {
        ReduceScatterMinimalDirectParams{
            scatter_dim,
            output_mem_config,
            cluster_axis,
            fabric_config,
            axis_topology,
            axis_num_devices,
            axis_num_links,
            num_devices,
            packet_size,
            resolved_num_links,
            subdevice_id,
            sub_core_grid},
        ReduceScatterMinimalDirectInputs{
            .input_tensor = input_tensor,
            .persistent_output_tensor = persistent_output_tensor,
            .persistent_staging_tensor = persistent_staging_tensor}};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> reduce_scatter_minimal_direct(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const ttnn::MemoryConfig& output_mem_config,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<ttnn::Tensor>& persistent_staging_tensor,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    auto [params, inputs] = ttnn::experimental::prim::build_operation_args(
        input_tensor,
        dim,
        output_mem_config,
        cluster_axis,
        num_links,
        persistent_output_tensor,
        persistent_staging_tensor,
        sub_device_id,
        sub_core_grid);
    return ttnn::device_operation::launch<ttnn::experimental::prim::ReduceScatterMinimalDirectDeviceOperation>(
        params, inputs);
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather_device_operation.hpp"
#include "high_bw_all_gather_device_operation_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/snake_ring.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

#include <algorithm>

namespace ttnn::operations::experimental::high_bw_all_gather {

namespace CMAKE_UNIQUE_NAMESPACE {

tt::tt_metal::Shape logical_input_shape(const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args) {
    auto shape = tensor_args.input_tensor.padded_shape();
    if (tensor_args.has_paged_input()) {
        shape[0] = 1;
        shape[1] = 1;
        shape[2] = tensor_args.page_bundle_indices->logical_volume() * args.kv_cache_page_size;
    }
    return shape;
}

void validate_paged_input(const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args) {
    if (!tensor_args.has_paged_input()) {
        return;
    }

    const auto& input = tensor_args.input_tensor;
    const auto& table = *tensor_args.page_bundle_indices;
    TT_FATAL(!args.input_batch_index.has_value(), "Paged input is incompatible with input_batch_index");
    TT_FATAL(args.dim == 2, "Paged high_bw_all_gather supports sequence gather on dim 2 only (got {})", args.dim);
    TT_FATAL(
        args.kv_cache_page_size >= tt::constants::TILE_HEIGHT &&
            args.kv_cache_page_size % tt::constants::TILE_HEIGHT == 0,
        "kv_cache_page_size must be a positive multiple of {} (got {})",
        tt::constants::TILE_HEIGHT,
        args.kv_cache_page_size);
    TT_FATAL(args.kv_cache_num_layers > 0, "kv_cache_num_layers must be positive");
    TT_FATAL(
        args.kv_cache_layer_idx < args.kv_cache_num_layers,
        "kv_cache_layer_idx {} must be less than kv_cache_num_layers {}",
        args.kv_cache_layer_idx,
        args.kv_cache_num_layers);

    TT_FATAL(table.storage_type() == StorageType::DEVICE, "page_bundle_indices must be on device");
    TT_FATAL(table.buffer() != nullptr, "page_bundle_indices must have an allocated device buffer");
    TT_FATAL(table.device() == input.device(), "page_bundle_indices must be on the same mesh device as input");
    TT_FATAL(table.dtype() == DataType::UINT16, "page_bundle_indices must have uint16 dtype");
    TT_FATAL(table.layout() == Layout::ROW_MAJOR, "page_bundle_indices must use ROW_MAJOR layout");
    TT_FATAL(table.padded_shape() == table.logical_shape(), "page_bundle_indices must not be padded");
    TT_FATAL(
        table.memory_config().buffer_type() == BufferType::DRAM && !table.memory_config().is_sharded(),
        "page_bundle_indices must be DRAM interleaved");
    const auto& table_shape = table.logical_shape();
    TT_FATAL(
        table_shape.rank() == 4 && table_shape[0] == 1 && table_shape[1] == 1 && table_shape[2] == 1 &&
            table_shape[3] > 0,
        "page_bundle_indices must have shape [1,1,1,num_logical_bundles] (got {})",
        table_shape);

    const auto& input_shape = input.logical_shape();
    TT_FATAL(
        input_shape.rank() == 4 && input_shape[1] == 1 && input_shape[2] == args.kv_cache_page_size,
        "Paged input must have shape [physical_bundles*num_layers,1,kv_cache_page_size,D] (got {})",
        input_shape);
    TT_FATAL(
        input_shape[0] > 0 && input_shape[0] % args.kv_cache_num_layers == 0,
        "Paged input flat page count {} must be positive and divisible by kv_cache_num_layers {}",
        input_shape[0],
        args.kv_cache_num_layers);
    const uint32_t physical_bundles = input_shape[0] / args.kv_cache_num_layers;
    TT_FATAL(
        physical_bundles <= (1u << 16),
        "uint16 page_bundle_indices support at most 65536 physical bundles (got {})",
        physical_bundles);
    TT_FATAL(input.memory_config().buffer_type() == BufferType::DRAM, "Paged input must be in DRAM");
    const auto nd = input.nd_shard_spec();
    TT_FATAL(nd.has_value(), "Paged input must use an ND-sharded memory config");
    const auto& shard = nd->shard_shape;
    TT_FATAL(
        shard.rank() == 4 && shard[0] == 1 && shard[1] == 1 && shard[2] == args.kv_cache_page_size &&
            shard[3] == input_shape[3],
        "Each paged input ND shard must be exactly [1,1,{},{}] (got {})",
        args.kv_cache_page_size,
        input_shape[3],
        shard);
}

tt::tt_metal::TensorTopology derive_output_topology(
    const Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis, uint32_t gather_dim) {
    const auto& input_topology = input_tensor.tensor_topology();
    auto output_placements = input_topology.placements();
    if (cluster_axis.has_value()) {
        TT_FATAL(
            *cluster_axis < output_placements.size() && *cluster_axis < input_topology.distribution_shape().dims(),
            "high_bw_all_gather requires the tensor distribution axes to match the device mesh axes; "
            "cluster_axis={}, distribution rank={}, placement count={}",
            *cluster_axis,
            input_topology.distribution_shape().dims(),
            output_placements.size());

        // Only the participating mesh axis becomes replicated. Another mesh axis may
        // independently shard the same tensor dimension and must remain sharded.
        output_placements[*cluster_axis] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
    } else {
        // A full-mesh ring consumes every shard placement of the gather dimension.
        const uint32_t rank = input_tensor.logical_shape().rank();
        for (auto& placement : output_placements) {
            if (ttnn::operations::ccl::common::placement_shards_tensor_dim(placement, gather_dim, rank)) {
                placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
            }
        }
    }
    return tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(output_placements), input_topology.mesh_coords());
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using namespace CMAKE_UNIQUE_NAMESPACE;

ttsl::hash::hash_t HighBwAllGatherDeviceOperation::compute_program_hash(
    const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args) {
    // input_batch_index / gathered_dim_size alter only runtime page ranges and iterator strides;
    // exclude their values so changing cache slot or prefix does not recompile the program.
    return tt::tt_metal::operation::hash_operation<HighBwAllGatherDeviceOperation>(
        args.dim,
        args.output_mem_config,
        args.cluster_axis,
        args.linearized_mesh_ring,
        args.snake_ring_orientation,
        args.fabric_config,
        args.axis_topology,
        args.axis_num_devices,
        args.axis_num_links,
        args.num_devices,
        args.num_links,
        args.mesh_rows,
        args.mesh_cols,
        args.packet_size,
        args.neighbor_unicast_eligible,
        args.neighbor_route_plan_hash,
        args.subdevice_id,
        args.sub_core_grid,
        args.input_batch_index.has_value(),
        args.gathered_dim_size.has_value(),
        args.kv_cache_page_size,
        args.kv_cache_num_layers,
        args.kv_cache_layer_idx,
        tensor_args);
}

void HighBwAllGatherDeviceOperation::validate_on_program_cache_miss(
    const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Constraints on input tensor
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffers on device!");

    // Constraints on other inputs
    int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    TT_FATAL(args.dim >= -rank && args.dim < rank, "Invalid gather dim {} for {}D input tensor", args.dim, rank);
    TT_FATAL(
        args.num_devices > 1,
        "high_bw_all_gather collective will only work for num_devices > 1, got {}",
        args.num_devices);

    // This op implements a single direct-neighbor line/ring. A full 2D mesh may
    // participate when it has been linearized into one Hamiltonian ring.
    const auto mesh_shape = input_tensor.device()->shape();
    if (args.linearized_mesh_ring) {
        const uint32_t snake_lane_count =
            args.snake_ring_orientation == ttnn::ccl::snake_ring::Orientation::Row ? mesh_shape[0] : mesh_shape[1];
        TT_FATAL(
            mesh_shape[0] > 1 && mesh_shape[1] > 1 && snake_lane_count % 2 == 0,
            "high_bw_all_gather full-mesh ring requires a 2D mesh whose selected snake orientation has an even "
            "lane count; mesh={}, orientation={}",
            mesh_shape,
            static_cast<uint32_t>(args.snake_ring_orientation));
        TT_FATAL(
            ttnn::operations::ccl::common::has_row_major_mesh_coordinates(input_tensor),
            "high_bw_all_gather full-mesh ring currently requires row-major tensor mesh coordinates");
        TT_FATAL(
            ttnn::operations::ccl::common::tensor_dim_shard_factor(input_tensor, args.dim) == args.num_devices,
            "high_bw_all_gather full-mesh ring requires the input to be sharded over gather dim {} across all {} "
            "devices",
            args.dim,
            args.num_devices);
    } else {
        TT_FATAL(
            args.cluster_axis < 2 && args.cluster_axis < mesh_shape.dims() && mesh_shape[args.cluster_axis] > 1,
            "high_bw_all_gather requires cluster_axis to select a mesh axis with at least two devices; "
            "cluster_axis={}, mesh_shape={}",
            args.cluster_axis,
            mesh_shape);
        TT_FATAL(
            (args.axis_num_devices[0] > 1) != (args.axis_num_devices[1] > 1),
            "high_bw_all_gather is a one-dimensional all-gather along cluster_axis {}; it cannot gather across "
            "both mesh axes",
            args.cluster_axis);
    }

    TT_FATAL(
        input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT || input_tensor.layout() == ttnn::TILE_LAYOUT,
        "high_bw_all_gather requires row-major or tile-layout input");
    TT_FATAL(input_tensor.buffer()->is_dram(), "high_bw_all_gather requires DRAM input");
    validate_paged_input(args, tensor_args);
    TT_FATAL(
        args.neighbor_unicast_eligible,
        "high_bw_all_gather requires a direct-neighbor line/ring; devices={}, links={}, topology={}, "
        "linearized_mesh_ring={}, route_hash={}",
        args.axis_num_devices,
        args.axis_num_links,
        args.axis_topology,
        args.linearized_mesh_ring,
        args.neighbor_route_plan_hash);

    {
        const auto& output_tensor = tensor_args.output_tensor;

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor must be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor must be allocated in buffers on device!");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(),
            "Output tensor layout {} should be same as input tensor layout {}",
            output_tensor.layout(),
            input_tensor.layout());
        TT_FATAL(
            output_tensor.dtype() == input_tensor.dtype(),
            "Output tensor dtype {} should be same as input tensor dtype {}",
            output_tensor.dtype(),
            input_tensor.dtype());
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Output tensor page config {} should be same as input tensor page config {}",
            output_tensor.tensor_spec().page_config(),
            input_tensor.tensor_spec().page_config());

        // Check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        auto input_padded_shape = logical_input_shape(args, tensor_args);
        auto expected_output_shape = input_padded_shape;
        expected_output_shape[args.dim] = input_padded_shape[args.dim] * args.num_devices;
        TT_FATAL(
            output_shape.size() == input_padded_shape.size(),
            "Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        if (args.input_batch_index.has_value()) {
            TT_FATAL(args.dim != 0, "high_bw_all_gather input_batch_index cannot be used when gathering dim 0");
            TT_FATAL(
                *args.input_batch_index < input_padded_shape[0],
                "high_bw_all_gather input_batch_index {} must be < input batch {}",
                *args.input_batch_index,
                input_padded_shape[0]);
            TT_FATAL(
                output_shape[0] == 1,
                "high_bw_all_gather selected-batch output must have batch 1, got {}",
                output_shape[0]);
            // The selected cache-slot path intentionally addresses one contiguous [1, ...] tensor.
            // Supporting arbitrary leading dimensions would make the selected page range discontinuous.
            for (int32_t i = 1; i < args.dim; ++i) {
                TT_FATAL(
                    input_padded_shape[i] == 1 && output_shape[i] == 1,
                    "high_bw_all_gather selected-batch path requires singleton dimensions between batch and dim; "
                    "input shape {}, output shape {}, dim {}",
                    input_padded_shape,
                    output_shape,
                    args.dim);
            }
            expected_output_shape[0] = 1;
        }
        TT_FATAL(
            output_shape == expected_output_shape,
            "Output tensor shape must be the worst-case gathered shape {}, got {}",
            expected_output_shape,
            output_shape);

        if (args.gathered_dim_size.has_value()) {
            const uint32_t gathered_dim_size = *args.gathered_dim_size;
            TT_FATAL(
                gathered_dim_size > 0 && gathered_dim_size <= expected_output_shape[args.dim],
                "high_bw_all_gather gathered_dim_size {} must be in (0, {}]",
                gathered_dim_size,
                expected_output_shape[args.dim]);
            TT_FATAL(
                gathered_dim_size % args.num_devices == 0,
                "high_bw_all_gather gathered_dim_size {} must divide evenly across {} devices",
                gathered_dim_size,
                args.num_devices);
        }
    }

    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            args.output_mem_config.buffer_type() == BufferType::DRAM,
        "high_bw_all_gather requires interleaved DRAM output");
}

void HighBwAllGatherDeviceOperation::validate_on_program_cache_hit(
    const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args) {
    // The slot/prefix values are deliberately hash-excluded. Recheck only their cheap dynamic
    // bounds here; all tensor/layout/fabric structure belongs to the program key and was proven on
    // the miss path. This keeps a serving-loop cache hit to scalar validation plus direct RT-arg writes.
    const auto input_shape = logical_input_shape(args, tensor_args);
    const auto& output_tensor = tensor_args.output_tensor;
    TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor must be allocated in buffers on device!");
    if (tensor_args.has_paged_input()) {
        TT_FATAL(
            tensor_args.page_bundle_indices->buffer() != nullptr,
            "page_bundle_indices must have an allocated device buffer");
    }
    if (args.input_batch_index.has_value()) {
        TT_FATAL(args.dim != 0, "high_bw_all_gather input_batch_index cannot be used when gathering dim 0");
        TT_FATAL(
            *args.input_batch_index < input_shape[0],
            "high_bw_all_gather input_batch_index {} must be < input batch {}",
            *args.input_batch_index,
            input_shape[0]);
    }
    if (args.gathered_dim_size.has_value()) {
        const int32_t rank = static_cast<int32_t>(input_shape.rank());
        const int32_t gather_dim = args.dim < 0 ? args.dim + rank : args.dim;
        const uint32_t max_gathered = input_shape[gather_dim] * args.num_devices;
        TT_FATAL(
            *args.gathered_dim_size > 0 && *args.gathered_dim_size <= max_gathered &&
                *args.gathered_dim_size % args.num_devices == 0,
            "high_bw_all_gather gathered_dim_size {} must be in (0, {}] and divide evenly across {} devices",
            *args.gathered_dim_size,
            max_gathered,
            args.num_devices);
    }
}

HighBwAllGatherDeviceOperation::spec_return_value_t HighBwAllGatherDeviceOperation::compute_output_specs(
    const HighBwAllGatherParams&, const HighBwAllGatherInputs& tensor_args) {
    // This op always writes the caller-provided persistent tensor.  In selected-batch mode its
    // batch dimension is intentionally smaller than the source cache, so its own spec—not one
    // re-derived from the input—is the authoritative output contract.
    return tensor_args.output_tensor.tensor_spec();
}

HighBwAllGatherDeviceOperation::topology_return_value_t HighBwAllGatherDeviceOperation::compute_output_topologies(
    const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args) {
    return {derive_output_topology(
        tensor_args.input_tensor,
        args.linearized_mesh_ring ? std::nullopt : std::optional<uint32_t>{args.cluster_axis},
        args.dim)};
}

HighBwAllGatherDeviceOperation::tensor_return_value_t HighBwAllGatherDeviceOperation::create_output_tensors(
    const HighBwAllGatherParams&, const HighBwAllGatherInputs& tensor_args) {
    return tensor_args.output_tensor;
}

HighBwAllGatherDeviceOperation::program_factory_t HighBwAllGatherDeviceOperation::select_program_factory(
    const HighBwAllGatherParams&, const HighBwAllGatherInputs&) {
    return program_factory_t{HighBwAllGatherUnicastFactory{}};
}

std::tuple<HighBwAllGatherParams, HighBwAllGatherInputs> high_bw_all_gather_build_operation_args(
    const Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    std::optional<uint32_t> num_links,
    std::optional<uint32_t> input_batch_index,
    std::optional<uint32_t> gathered_dim_size,
    const std::optional<Tensor>& page_bundle_indices,
    uint32_t kv_cache_page_size,
    uint32_t kv_cache_num_layers,
    uint32_t kv_cache_layer_idx) {
    // Query the machine and Fabric setup info.
    // This info is also effectively part of CCL args and hence should be in the program-cache hash,
    // so we include it in HighBwAllGatherParams.
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Input tensor should be on device for high_bw_all_gather operation");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(
        mesh_shape.dims() == 2, "high_bw_all_gather requires a two-dimensional device mesh shape, got {}", mesh_shape);
    const bool linearized_mesh_ring = !cluster_axis.has_value();
    if (cluster_axis.has_value()) {
        TT_FATAL(
            *cluster_axis < 2 && *cluster_axis < mesh_shape.dims(),
            "high_bw_all_gather cluster_axis must be 0, 1, or None; got {} for mesh shape {}",
            *cluster_axis,
            mesh_shape);
        TT_FATAL(
            mesh_shape[*cluster_axis] > 1,
            "high_bw_all_gather cluster_axis {} selects a singleton mesh axis in mesh shape {}",
            *cluster_axis,
            mesh_shape);
    } else {
        TT_FATAL(
            mesh_shape[0] > 1 && mesh_shape[1] > 1 && (mesh_shape[0] % 2 == 0 || mesh_shape[1] % 2 == 0),
            "high_bw_all_gather cluster_axis=None requires a 2D mesh with at least one even dimension, got {}",
            mesh_shape);
    }
    TT_FATAL(
        output_tensor.device() == mesh_device,
        "high_bw_all_gather input and output tensors must be on the same mesh device");

    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    // Axis 0 is N/S, and axis 1 is E/W.
    // An inactive axis has num_devices = 1, num_links = 0, Linear topology.
    std::array<tt::tt_fabric::Topology, 2> axis_topology{
        tt::tt_fabric::Topology::Linear, tt::tt_fabric::Topology::Linear};
    std::array<uint32_t, 2> axis_num_devices{1u, 1u};
    std::array<uint32_t, 2> axis_num_links{0u, 0u};
    std::optional<uint32_t> resolved_num_links;
    for (uint32_t axis = 0; axis < 2; ++axis) {
        const bool is_axis_active = mesh_shape[axis] > 1 && (linearized_mesh_ring || *cluster_axis == axis);
        if (!is_axis_active) {
            continue;
        }
        axis_topology[axis] = ::ttnn::ccl::get_axis_topology(input_tensor, fabric_config, axis);
        axis_num_devices[axis] = ::ttnn::ccl::get_topological_dimension(input_tensor, axis);
        const auto discovered_num_links =
            static_cast<uint32_t>(ttnn::operations::ccl::common::get_num_links(*mesh_device, axis));
        if (num_links.has_value()) {
            TT_FATAL(
                *num_links > 0,
                "high_bw_all_gather num_links must be greater than 0 when specified, got {}",
                *num_links);
            TT_FATAL(
                *num_links <= discovered_num_links,
                "high_bw_all_gather requested {} links, but only {} usable links were discovered on cluster_axis {}",
                *num_links,
                discovered_num_links,
                axis);
        }
        axis_num_links[axis] = num_links.value_or(discovered_num_links);
        resolved_num_links =
            resolved_num_links.has_value() ? std::min(*resolved_num_links, axis_num_links[axis]) : axis_num_links[axis];
    }
    TT_FATAL(resolved_num_links.has_value(), "high_bw_all_gather found no active collective axis");
    const uint32_t collective_num_links = *resolved_num_links;
    const uint32_t num_devices = linearized_mesh_ring ? static_cast<uint32_t>(mesh_shape.mesh_size())
                                                      : axis_num_devices[0] * axis_num_devices[1];
    const size_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const bool one_active_axis = (axis_num_devices[0] > 1) != (axis_num_devices[1] > 1);
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(fabric_config);
    ttnn::ccl::snake_ring::Orientation snake_orientation = linearized_mesh_ring && mesh_shape[0] % 2 != 0
                                                               ? ttnn::ccl::snake_ring::Orientation::Column
                                                               : ttnn::ccl::snake_ring::Orientation::Row;
    std::optional<uint64_t> direct_neighbor_route_hash;
    if (fabric_is_2d && (linearized_mesh_ring || one_active_axis)) {
        const auto mesh_ring_plan = ttnn::operations::ccl::common::resolve_mesh_ring_plan(
            input_tensor, cluster_axis, collective_num_links, axis_topology, true, "high_bw_all_gather");
        if (mesh_ring_plan.has_value()) {
            snake_orientation = mesh_ring_plan->orientation;
            direct_neighbor_route_hash = mesh_ring_plan->route_plan_hash;
        }
    }
    const uint32_t active_axis = cluster_axis.value_or(0);
    const auto active_topology = axis_topology[active_axis];
    const bool topology_supports_neighbor_unicast =
        active_topology == tt::tt_fabric::Topology::Linear || active_topology == tt::tt_fabric::Topology::Ring;
    // A direct physical line or ring is supported. The full edge proof keeps routed multi-hop logical rings out of
    // this native one-hop implementation.
    const bool neighbor_unicast_eligible =
        collective_num_links > 0 &&
        ((!linearized_mesh_ring && one_active_axis && !fabric_is_2d && topology_supports_neighbor_unicast) ||
         (fabric_is_2d && direct_neighbor_route_hash.has_value()));

    log_debug(
        tt::LogOp,
        "fabric_config: {}, axis_topology: {}, axis_num_devices: {}, axis_num_links: {}, num_links override: {}, "
        "linearized_mesh_ring: {}, snake_orientation: {}, packet_size: {} B",
        fabric_config,
        axis_topology,
        axis_num_devices,
        axis_num_links,
        num_links,
        linearized_mesh_ring,
        static_cast<uint32_t>(snake_orientation),
        packet_size);

    // Resolve negative gather dim
    uint32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    return {
        HighBwAllGatherParams{
            .dim = gather_dim,
            .output_mem_config = output_tensor.memory_config(),
            .cluster_axis = cluster_axis.value_or(0),
            .linearized_mesh_ring = linearized_mesh_ring,
            .snake_ring_orientation = snake_orientation,
            .fabric_config = fabric_config,
            .axis_topology = axis_topology,
            .axis_num_devices = axis_num_devices,
            .axis_num_links = axis_num_links,
            .num_devices = num_devices,
            .num_links = collective_num_links,
            .mesh_rows = linearized_mesh_ring ? mesh_shape[0] : 0,
            .mesh_cols = linearized_mesh_ring ? mesh_shape[1] : 0,
            .packet_size = packet_size,
            .neighbor_unicast_eligible = neighbor_unicast_eligible,
            .neighbor_route_plan_hash = direct_neighbor_route_hash,
            .subdevice_id = subdevice_id,
            .sub_core_grid = sub_core_grid,
            .input_batch_index = input_batch_index,
            .gathered_dim_size = gathered_dim_size,
            .kv_cache_page_size = kv_cache_page_size,
            .kv_cache_num_layers = kv_cache_num_layers,
            .kv_cache_layer_idx = kv_cache_layer_idx},
        HighBwAllGatherInputs{
            .input_tensor = input_tensor, .output_tensor = output_tensor, .page_bundle_indices = page_bundle_indices}};
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    std::optional<uint32_t> num_links,
    std::optional<uint32_t> input_batch_index,
    std::optional<uint32_t> gathered_dim_size,
    const std::optional<Tensor>& page_bundle_indices,
    uint32_t kv_cache_page_size,
    uint32_t kv_cache_num_layers,
    uint32_t kv_cache_layer_idx) {
    auto [params, inputs] = ttnn::operations::experimental::high_bw_all_gather::high_bw_all_gather_build_operation_args(
        input_tensor,
        output_tensor,
        dim,
        cluster_axis,
        subdevice_id,
        sub_core_grid,
        num_links,
        input_batch_index,
        gathered_dim_size,
        page_bundle_indices,
        kv_cache_page_size,
        kv_cache_num_layers,
        kv_cache_layer_idx);
    return ttnn::device_operation::launch<
        ttnn::operations::experimental::high_bw_all_gather::HighBwAllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn::prim

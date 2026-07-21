// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_device_operation.hpp"
#include "all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::ccl {

void AllGatherDeviceOperation::validate_on_program_cache_miss(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Constraints on input tensor
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffers on device!");

    // Constraints on other inputs
    int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    TT_FATAL(args.dim >= -rank && args.dim < rank, "Invalid gather dim {} for {}D input tensor", args.dim, rank);
    TT_FATAL(
        args.num_devices > 1, "all_gather collective will only work for num_devices > 1, got {}", args.num_devices);

    // If mesh_device shape is 2D but !FABRIC_2D, then must specify cluster_axis
    const auto mesh_shape = input_tensor.device()->shape();
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(args.fabric_config);
    TT_FATAL(
        fabric_is_2d || args.cluster_axis.has_value() || mesh_shape[0] == 1 || mesh_shape[1] == 1,
        "1D fabric on a 2D mesh_device requires cluster_axis to be set");

    // Constraints on persistent output tensor
    if (tensor_args.persistent_output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.persistent_output_tensor.value();

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor must be on device!");
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
        auto input_shape = input_tensor.padded_shape();
        auto expected_output_shape = input_shape;
        expected_output_shape[args.dim] *= args.num_devices;
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        TT_FATAL(
            output_shape == expected_output_shape,
            "Output tensor shape must be {}, got {}",
            expected_output_shape,
            output_shape);
    }

    // NOTE: implementation-specific constraints (row-major page alignment, tile
    // padding on the gather dim, input/output page-size ratios, concat vs multi-shard,
    // etc.) are not checked here. use_composite_all_gather() checks those and routes
    // to composite path if needed.
}

AllGatherDeviceOperation::spec_return_value_t AllGatherDeviceOperation::compute_output_specs(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] *= args.num_devices;
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

AllGatherDeviceOperation::topology_return_value_t AllGatherDeviceOperation::compute_output_topologies(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();
    auto output_placements = input_topology.placements();

    // For each distribution dimension, if sharded on the gather dim, make it replicated
    for (auto& output_placement : output_placements) {
        if (auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&output_placement)) {
            if (shard->dim == static_cast<int>(args.dim)) {
                output_placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
            }
        }
    }

    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), output_placements, input_topology.mesh_coords())};
}

AllGatherDeviceOperation::tensor_return_value_t AllGatherDeviceOperation::create_output_tensors(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    if (tensor_args.persistent_output_tensor.has_value()) {
        return tensor_args.persistent_output_tensor.value();
    }
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<AllGatherDeviceOperation::tensor_return_value_t>
AllGatherDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    // =========================================================================
    // AllGather roofline performance model
    //
    // Fabric perf (bandwidth and latency):
    //   bandwidth: the worst-connected device must still receieve the (N-1) slices
    //       it lacks, i.e. (N-1)*S bytes, pulled in over its K inbound links. A
    //       wrapped axis (Ring/Torus) gives that node 2 links on that axis, else 1.
    //   latency: the gather finishes only once the farthest slice has crossed the
    //       network diameter; for a grid/torus that is the sum of per-axis diameters.
    // Computed per-axis, so line / ring / mesh / torus_x / torus_y / torus_xy all
    // share one code path.
    //
    // DRAM perf (memory bandwidth ceiling):
    // Each device reads S bytes and writes N*S bytes.
    // =========================================================================

    const auto& input_tensor = tensor_args.input_tensor;

    // Architecture and clock detection
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    int clock_rate_mhz = 1000;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        arch = input_tensor.device()->arch();
        clock_rate_mhz = input_tensor.device()->get_clock_rate_mhz();
    }

    // Data size: bytes each device contributes
    const uint64_t input_size_bytes = input_tensor.physical_volume() * input_tensor.element_size();

    const uint32_t num_devices = args.num_devices;

    // --- Fabric roofline: fold every axis into K (cabling) and the diameter (hops). ---
    // K    = inbound links at the worst-connected node: wrapped axis -> 2 links, open -> 1.
    // hops = network diameter = sum of per-axis diameters.
    // Computed per-axis, so line/ring/mesh/torus_x/torus_y/torus_xy share same formula.
    uint32_t bottleneck_links = 0;
    uint32_t diameter_hops = 0;
    for (size_t axis = 0; axis < args.axis_num_devices.size(); ++axis) {
        const uint32_t axis_devices = args.axis_num_devices[axis];
        if (axis_devices <= 1) {
            continue;  // inactive axis: no links, no hops (reduces to 1D)
        }
        const bool axis_wraps = tt::tt_fabric::is_ring_or_torus(args.axis_topology[axis]);
        bottleneck_links += (axis_wraps ? 2u : 1u) * args.axis_num_links[axis];
        diameter_hops += axis_wraps ? (axis_devices / 2u) : (axis_devices - 1u);
    }

    // (N-1)*S bytes must be received over K links, farthest byte travels `diameter_hops` hops.
    const uint64_t fabric_bytes = static_cast<uint64_t>(num_devices - 1) * input_size_bytes;
    const auto [fabric_bw_cycles, fabric_fill_cycles] = ttnn::ccl::estimate_fabric_transfer_cycles(
        arch, args.fabric_config, clock_rate_mhz, fabric_bytes, bottleneck_links, diameter_hops);

    // --- Local DRAM bandwidth ceiling (first-principles) ---
    // Read: device reads S bytes from DRAM.  Write: device writes N*S bytes.
    // Roofline assumes all device compute cores drive DRAM concurrently
    // (hardware max parallelism). DRAM terms compete (overlap) with Fabric.
    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const uint32_t read_page_size = input_tensor.buffer()->page_size();
    const uint32_t write_page_size = output_tensor.buffer()->page_size();

    // Hardware ceiling: total compute cores available to drive DRAM
    const uint32_t device_rows = input_tensor.device()->compute_with_storage_grid_size().x;
    const uint32_t device_cols = input_tensor.device()->compute_with_storage_grid_size().y;
    const uint32_t num_cores = device_rows * device_cols;

    const int64_t output_size_bytes = num_devices * input_size_bytes;
    const uint32_t read_pages = tt::div_up(input_size_bytes, read_page_size);
    const uint32_t write_pages = tt::div_up(output_size_bytes, write_page_size);
    const uint32_t read_pages_per_core = tt::div_up(read_pages, num_cores);
    const uint32_t write_pages_per_core = tt::div_up(write_pages, num_cores);

    auto [read_bw_cycles, read_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        read_page_size, input_is_dram, /*is_local=*/false, read_pages_per_core, arch, /*is_read=*/true);
    auto [write_bw_cycles, write_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        write_page_size, output_is_dram, /*is_local=*/false, write_pages_per_core, arch, /*is_read=*/false);

    const int local_bw_cycles = static_cast<int>(std::max(read_bw_cycles, write_bw_cycles));
    const int pipeline_latency_cycles = static_cast<int>(read_latency_cycles + write_latency_cycles);

    // Throughput and latency don't add, they overlap; since the ingest link carries data from nearby
    // devices (throughput) while in parallel the farthest data is travelling over the network (latency).
    const int throughput_cycles = std::max(local_bw_cycles, fabric_bw_cycles);
    const int fill_cycles = std::max(pipeline_latency_cycles, fabric_fill_cycles);
    const int ideal_dev_clock_cycles = std::max(throughput_cycles, fill_cycles);

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

AllGatherDeviceOperation::program_factory_t AllGatherDeviceOperation::select_program_factory(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    // Heuristics to pick the kernel algorithm.
    // Multicast supports all Fabric topologies, unicast only supports Fabric 1D topologies.
    // Unicast is empirically found to be faster for large tensors.
    bool use_unicast = false;
    if (args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE) {
        // NeighborExchange only permits 1-hop unicast
        use_unicast = true;
    } else if (!::tt::tt_fabric::is_2d_fabric_config(args.fabric_config)) {
        // Unicast is only supported for Fabric 1D configs
        const auto& input_tensor = tensor_args.input_tensor;
        switch (input_tensor.device()->arch()) {
            case tt::ARCH::WORMHOLE_B0: {
                const uint64_t num_pages = input_tensor.buffer()->num_pages();       // per-device shard
                const bool large_page = input_tensor.buffer()->page_size() >= 4096;  // fp32 / int32 / wide row-major
                if (args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_RING) {
                    use_unicast = num_pages >= (large_page ? 20u : 64u);  // large pages cross far earlier
                }
                break;
            }
            case tt::ARCH::BLACKHOLE: {
                // Multicast only wins the small-volume + large-page corner; unicast is
                // bandwidth-optimal and wins outright once ~4MB cross a link (at any page size).
                // Between, the multicast-favorable per-link-byte ceiling scales with the actual page
                // size squared, up to that 4MB floor. The boundary errs toward unicast: its downside
                // (up to ~40% at scale) dwarfs multicast's ~5-17% upside.
                const uint64_t in_page = input_tensor.tensor_spec().compute_page_size_bytes();
                const uint64_t out_page = compute_output_specs(args, tensor_args).compute_page_size_bytes();
                const uint64_t txn = std::min(in_page, out_page);  // NOC transaction size

                // per-link bytes on the gathered axis (same axis/links the unicast factory uses)
                const uint32_t axis = args.cluster_axis.value_or(args.axis_num_devices[1] > 1 ? 1 : 0);  // max 2 axes
                const uint64_t num_links = std::max<uint32_t>(1u, args.axis_num_links[axis]);
                const uint64_t per_link_bytes =
                    input_tensor.physical_volume() * input_tensor.element_size() * args.num_devices / num_links;

                // Page^2 ceiling, anchored so a 2KB page permits multicast up to 1MB/link, capped at
                // unicast's 4MB bandwidth floor. (2KB/1MB are the anchor; txn is the real page size.)
                constexpr uint64_t anchor_page = 2048, anchor_ceiling = 1'000'000, unicast_bw_floor = 4'000'000;
                const uint64_t mcast_ceiling =
                    std::min(unicast_bw_floor, (txn * txn * anchor_ceiling) / (anchor_page * anchor_page));
                use_unicast = per_link_bytes >= mcast_ceiling;
                break;
            }
            default: break;  // uncalibrated arch
        }
    }

    return use_unicast ? program_factory_t{AllGatherUnicastFactory{}} : program_factory_t{AllGatherMulticastFactory{}};
}

std::tuple<AllGatherParams, AllGatherInputs> all_gather_build_operation_args(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // Query the machine and Fabric setup info.
    // This info is also effectively part of CCL args and hence should be in the program-cache hash,
    // so we include it in AllGatherParams.
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Input tensor should be on device for all_gather operation");
    const auto mesh_shape = mesh_device->shape();
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    // Axis 0 is N/S, and axis 1 is E/W.
    // An inactive axis has num_devices = 1, num_links = 0, Linear topology.
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
    const uint32_t num_devices = axis_num_devices[0] * axis_num_devices[1];  // devices partaking in the collective
    const size_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    log_debug(
        tt::LogOp,
        "fabric_config: {}, axis_topology: {}, axis_num_devices: {}, axis_num_links: {}, packet_size: {} B",
        fabric_config,
        axis_topology,
        axis_num_devices,
        axis_num_links,
        packet_size);

    // Resolve negative gather dim
    uint32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    return {
        AllGatherParams{
            gather_dim,
            memory_config.value_or(input_tensor.memory_config()),
            cluster_axis,
            fabric_config,
            axis_topology,
            axis_num_devices,
            axis_num_links,
            num_devices,
            packet_size,
            subdevice_id,
            sub_core_grid},
        AllGatherInputs{.input_tensor = input_tensor, .persistent_output_tensor = persistent_output_tensor}};
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {

Tensor all_gather(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    auto [params, inputs] = ttnn::operations::ccl::all_gather_build_operation_args(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid);
    return ttnn::device_operation::launch<ttnn::operations::ccl::AllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn::prim

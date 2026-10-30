// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_device_operation.hpp"
#include "all_gather_device_operation_types.hpp"
#include "all_gather_test_knobs.hpp"  // TEST ONLY

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
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
    const int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    TT_FATAL(
        args.dim_from_end >= -rank && args.dim_from_end < 0,
        "Invalid (normalized) gather dim {} for {}D input tensor",
        rank + args.dim_from_end,
        rank);
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
        expected_output_shape[args.dim_from_end] *= args.num_devices;
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
    const AllGatherParams& args, const AllGatherInputs& /*tensor_args*/) {
    return args.output_spec;
}

AllGatherDeviceOperation::topology_return_value_t AllGatherDeviceOperation::compute_output_topologies(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    // The gathered mesh axis becomes Replicate; the other axes keep their placements even when they shard the
    // gathered tensor dim (their pieces are still distinct after the gather). Without a cluster_axis the gather
    // spans the whole mesh and every placement becomes Replicate. The helper expands a collapsed 1-D label to one
    // placement per mesh axis first and compares Shard dims normalised, treating an out-of-range dim left behind by
    // a rank-changing op (#52331) as matching nothing. `dim_from_end` is negative by construction; the helper
    // normalises it. No honest label (nullopt, already
    // reported): the output keeps the input's label. Returning {} would instead hand the framework the union over
    // every tensor argument, the persistent output tensor's label included.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto output_topology =
        ttnn::operations::ccl::common::all_gather_output_topology(input_tensor, args.cluster_axis, args.dim_from_end);
    return {output_topology.value_or(input_tensor.tensor_topology())};
}

AllGatherDeviceOperation::tensor_return_value_t AllGatherDeviceOperation::create_output_tensors(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    if (tensor_args.persistent_output_tensor.has_value()) {
        return tensor_args.persistent_output_tensor.value();
    }
    return create_device_tensor(args.output_spec, tensor_args.input_tensor.device());
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

namespace {

// A sharded output whose shards go round-robin to the banks puts every device's stripe in one bank when a stripe is
// within one shard along the gather dim and every other shard-grid dim steps by whole rounds of the banks.
bool stripe_in_one_bank(const tt::tt_metal::TensorSpec& output_spec, int32_t dim_from_end, uint32_t num_devices) {
    const auto& nd_shard_spec = output_spec.memory_config().nd_shard_spec();
    const auto& shape = output_spec.padded_shape();
    if (!nd_shard_spec.has_value() || nd_shard_spec->shard_shape.rank() != shape.rank() ||
        nd_shard_spec->shard_distribution_strategy != tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D) {
        return false;
    }
    const auto& shard_shape = nd_shard_spec->shard_shape;
    const uint64_t num_banks = nd_shard_spec->grid.num_cores();
    const int32_t gather_dim = static_cast<int32_t>(shape.rank()) + dim_from_end;
    if (shard_shape[gather_dim] % (shape[gather_dim] / num_devices) != 0) {
        return false;
    }
    uint64_t stride = 1;  // shards between neighbours along dim i
    for (int32_t i = static_cast<int32_t>(shape.rank()) - 1; i >= 0; --i) {
        const uint64_t shards = tt::div_up(shape[i], shard_shape[i]);
        if (i != gather_dim && shards > 1 && stride % num_banks != 0) {
            return false;
        }
        stride *= shards;
    }
    return true;
}

}  // namespace

AllGatherDeviceOperation::program_factory_t AllGatherDeviceOperation::select_program_factory(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    // Pick the kernel algorithm based on limitations and heuristics
    bool use_unicast = false;
    bool heuristic = false;  // TEST ONLY
    // The factories build for the output tensor itself, and a persistent one may live in another memory.
    const auto& output_spec = tensor_args.persistent_output_tensor.has_value()
                                  ? tensor_args.persistent_output_tensor->tensor_spec()
                                  : args.output_spec;
    if (args.is_true_2d()) {
        // Limitation: Unicast algorithm currently does not support true 2D topologies
        use_unicast = false;
        // NOLINTNEXTLINE(bugprone-branch-clone) - one branch per limitation, kept separate to document each
    } else if (tt::tt_fabric::is_2d_fabric_config(args.fabric_config) && !args.axis_is_straight[args.get_1d_axis()]) {
        // Limitation: Multicast cannot handle 2D topology reshapes/views that bend an axis
        use_unicast = true;
    } else if (args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE) {
        // Limitation: NeighborExchange only permits 1-hop unicast
        use_unicast = true;
    } else if (get_available_worker_cores(args, tensor_args.input_tensor).num_cores() < 4) {
        // Limitation: unicast needs a fabric mux and a worker per direction
        use_unicast = false;
    } else if (!compute_walk_geometry(
                    tensor_args.input_tensor,
                    output_spec,
                    args.dim_from_end,
                    args.num_devices,
                    args.packet_size,
                    /*unicast=*/true)
                    .has_value()) {
        // Limitation: the unicast relay re-reads its own output, so pages resharded on the way (concat or split)
        // must also be cut on the output's read alignment
        use_unicast = false;
    } else {
        heuristic = true;  // TEST ONLY: TT_AG_FORCE may override from here on
        // Heuristics: Decide between multicast or unicast algorithm
        const auto& input_tensor = tensor_args.input_tensor;
        const uint32_t axis = args.get_1d_axis();
        switch (input_tensor.device()->arch()) {
            case tt::ARCH::WORMHOLE_B0: {
                const uint64_t num_pages = input_tensor.buffer()->num_pages();       // per-device shard
                const bool large_page = input_tensor.buffer()->page_size() >= 4096;  // fp32 / int32 / wide row-major
                if (tt::tt_fabric::is_ring_or_torus(args.axis_topology[axis])) {
                    use_unicast = num_pages >= (large_page ? 20u : 64u);  // large pages cross far earlier
                }
                break;
            }
            case tt::ARCH::BLACKHOLE: {
                // Multicast sends each packet once, unicast relays it hop by hop at a higher bandwidth but with more
                // cores to start (workers and a mux per link-direction): unicast wins once enough bytes cross the
                // busiest link-direction. Measured crossover (2 links, 15168 B payload, bf16 tiles 64 KB - 42 MB,
                // init handshake included): 1.05 MB on a line, 233 KB on a ring, where both directions carry it.
                const bool is_ring = tt::tt_fabric::is_ring_or_torus(args.axis_topology[axis]);
                const uint64_t num_links = std::max<uint32_t>(1u, args.axis_num_links[axis]);
                const uint64_t input_bytes =
                    static_cast<uint64_t>(input_tensor.buffer()->num_pages()) * input_tensor.buffer()->page_size();
                const uint64_t busiest_link_bytes =
                    input_bytes * (args.num_devices - 1) / ((is_ring ? 2u : 1u) * num_links);
                constexpr uint64_t line_unicast_bytes = 1'050'000, ring_unicast_bytes = 233'000;
                use_unicast = busiest_link_bytes >= (is_ring ? ring_unicast_bytes : line_unicast_bytes);
                // Except when every device's output stripe sits in one bank: the unicast relay's writes and re-reads
                // all queue on it (ND-sharded DRAM, 8 MB: multicast 1.3-1.4x faster).
                use_unicast = use_unicast && !stripe_in_one_bank(output_spec, args.dim_from_end, args.num_devices);
                break;
            }
            default: break;  // uncalibrated arch
        }
    }
    if (const auto force = ag_test::env_str("TT_AG_FORCE")) {  // TEST ONLY
        TT_FATAL(*force == "unicast" || *force == "multicast", "TT_AG_FORCE must be unicast or multicast");
        TT_FATAL(
            heuristic || use_unicast == (*force == "unicast"),
            "TT_AG_FORCE={}: a legality rule requires {} here (true 2D, bent 2D axis, neighbor exchange, < 4 cores, or "
            "no unicast walk geometry)",
            *force,
            use_unicast ? "unicast" : "multicast");
        use_unicast = *force == "unicast";
    }

    return use_unicast ? program_factory_t{AllGatherUnicastFactory{}} : program_factory_t{AllGatherMulticastFactory{}};
}

tt::tt_metal::TensorSpec compute_output_specs_helper(
    const Tensor& input_tensor,
    int32_t gather_dim_from_end,
    uint32_t num_devices,
    const std::optional<MemoryConfig>& memory_config) {
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
    if (output_mem_config.created_with_nd_shard_spec()) {
        // Forward only the nd_shard_spec: the legacy 2D spec that TensorSpec back-fills describes the
        // *input* shape, and its one-shard-per-core rule rejects valid ND configs for the larger output.
        output_mem_config = MemoryConfig(output_mem_config.buffer_type(), output_mem_config.nd_shard_spec());
    }
    auto shape = input_tensor.logical_shape();
    shape[gather_dim_from_end] *= num_devices;
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config));
}

std::optional<all_gather_walk::WalkGeo> compute_walk_geometry(
    const Tensor& input_tensor,
    const tt::tt_metal::TensorSpec& output_spec,
    int32_t gather_dim_from_end,
    uint32_t num_devices,
    uint32_t packet_size,
    bool unicast) {
    const auto& input_spec = input_tensor.tensor_spec();
    auto read_alignment = [](const tt::tt_metal::TensorSpec& spec) {
        return spec.memory_config().is_dram() ? tt::tt_metal::hal::get_dram_alignment()
                                              : tt::tt_metal::hal::get_l1_alignment();
    };
    // Page sizes are UNALIGNED (content) sizes.
    const uint32_t input_page_size = input_spec.compute_page_size_bytes();
    const uint32_t output_page_size = output_spec.compute_page_size_bytes();
    const uint32_t input_aligned_page_size = tt::align(input_page_size, read_alignment(input_spec));
    const uint32_t output_aligned_page_size = tt::align(output_page_size, read_alignment(output_spec));

    // --- Unit ---
    // Every source is read at unit offsets into a unit-aligned slot, so the unit must be a multiple of each source's
    // read alignment (else BH reads (src & ~63) | (dst & 63)) and of the NOC write alignment (the L1 alignment).
    const uint32_t unit_bytes = std::lcm(
        std::lcm(tt::tt_metal::hal::get_l1_alignment(), read_alignment(input_spec)),
        unicast ? read_alignment(output_spec) : 1u);
    uint32_t in_page_units = 0;
    uint32_t out_page_units = 0;
    if (input_page_size % unit_bytes == 0 && output_page_size % unit_bytes == 0) {
        // Cuts land on any unit boundary, also inside a page (concat and split).
        in_page_units = input_page_size / unit_bytes;
        out_page_units = output_page_size / unit_bytes;
    } else if (
        input_page_size == output_page_size && tt::align(input_page_size, unit_bytes) <= output_aligned_page_size) {
        // Whole pages move at pitch align(content, unit); the tail lands in the output page's own padding.
        in_page_units = out_page_units = tt::align(input_page_size, unit_bytes) / unit_bytes;
    } else {
        return std::nullopt;
    }
    const uint32_t max_packet_bytes =
        std::min<uint32_t>(packet_size, tt::tt_metal::hal::get_noc_max_burst_size_bytes()) / unit_bytes * unit_bytes;
    if (max_packet_bytes == 0) {
        return std::nullopt;
    }

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather dim .. last dim] this device contributes per stripe. For
    // a last-dim RM gather this is the *page* count, which handles sharded RM input (> 1 input page per row).
    const auto& input_shape = input_tensor.padded_shape();
    const bool is_tile = input_tensor.layout() == ttnn::TILE_LAYOUT;
    auto tile_spec = is_tile ? input_spec.tile() : tt::tt_metal::Tile();
    uint64_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim_from_end; i < 0; i++) {
        uint64_t extent;
        if (i == -1) {
            if (is_tile) {
                extent = input_shape[i] / tile_spec.get_width();
            } else {
                // This is a page count, so divide by the unaligned page size, not aligned
                const uint64_t row_bytes = uint64_t{input_shape[i]} * input_tensor.element_size();
                if (row_bytes % input_page_size != 0) {
                    return std::nullopt;
                }
                extent = row_bytes / input_page_size;
            }
        } else if (is_tile && i == -2) {
            extent = input_shape[i] / tile_spec.get_height();
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }
    const uint64_t num_input_pages = input_tensor.buffer()->num_pages();
    if (input_pages_per_stripe == 0 || num_input_pages % input_pages_per_stripe != 0) {
        return std::nullopt;
    }
    // Stripe = this device's contribution to one output row, in units. A walk element is one output page when the
    // stripe is whole output pages, else one unit.
    const uint64_t stripe_units = input_pages_per_stripe * in_page_units;
    const uint32_t element_units = stripe_units % out_page_units == 0 ? out_page_units : 1u;
    if (stripe_units * num_devices > std::numeric_limits<uint32_t>::max() ||
        num_input_pages * in_page_units > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return all_gather_walk::WalkGeo{
        .unit_bytes = unit_bytes,
        .in_page_units = in_page_units,
        .out_page_units = out_page_units,
        .stripe_units = static_cast<uint32_t>(stripe_units),
        .num_devices = num_devices,
        .element_units = element_units,
        .max_packet_bytes = max_packet_bytes,
        .dst_cross = out_page_units * unit_bytes == output_aligned_page_size,
        .src_cross = in_page_units * unit_bytes == input_aligned_page_size};
}

uint32_t compute_packet_cap(const all_gather_walk::WalkGeo& geo, const Tensor& output_tensor) {
    // The output's page stride between banks and its contiguous runs bound how long a packet segment can be.
    const auto* buffer = output_tensor.buffer();
    const auto& dist_spec = buffer->buffer_distribution_spec();
    const uint32_t page_stride = dist_spec.has_value()
                                     ? dist_spec->contiguous_page_stride()
                                     : output_tensor.device()->allocator()->get_num_banks(buffer->buffer_type());
    const uint32_t run_pages =
        dist_spec.has_value() ? dist_spec->num_contiguous_pages(0) : std::numeric_limits<uint32_t>::max();
    return all_gather_walk::packet_cap(geo, page_stride, run_pages);
}

uint32_t compute_packets_per_slot(
    const all_gather_walk::WalkGeo& geo,
    uint32_t packet_cap,
    uint32_t num_slices,
    uint32_t header_bytes,
    const Tensor& input_tensor,
    uint32_t min_slots) {
    // A slot's packets are read behind one read barrier: enough of them to keep about 30 KB of reads in flight, which
    // hides DRAM read latency whatever the fabric payload (BH: two 15168 B packets per barrier take 24% less worker
    // time per packet than one), at least min_slots slots per worker iteration, and only as many as fit in L1.
    const uint32_t read_bytes_per_barrier = ag_test::env_u32("TT_AG_READ_BYTES").value_or(30336);  // TEST ONLY env
    constexpr uint32_t max_packets_per_slot = 32;
    const uint32_t num_elements = input_tensor.buffer()->num_pages() * geo.in_page_units / geo.element_units;
    const uint64_t iter_packets = tt::div_up(
        uint64_t{tt::div_up(num_elements, num_slices)} * geo.element_units * geo.unit_bytes, uint64_t{packet_cap});
    uint32_t packets_per_slot = static_cast<uint32_t>(std::clamp<uint64_t>(
        std::min<uint64_t>(tt::div_up(read_bytes_per_barrier, packet_cap), iter_packets / min_slots),
        1,
        max_packets_per_slot));
    if (const auto k = ag_test::env_u32("TT_AG_K")) {  // TEST ONLY
        packets_per_slot = std::max(*k, 1u);
    }
    auto cb_bytes = [&](uint32_t k) {
        return all_gather_walk::kNumSlots *
               all_gather_walk::slot_offsets(k, geo.unit_bytes, header_bytes, packet_cap).slot_bytes;
    };
    const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
    if (packets_per_slot > 1 && cb_bytes(packets_per_slot) > max_l1_space) {
        while (packets_per_slot > 1 && cb_bytes(packets_per_slot) > max_l1_space) {
            --packets_per_slot;
        }
        log_warning(
            tt::LogOp,
            "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
            max_l1_space);
    }
    return packets_per_slot;
}

CoreRangeSet get_available_worker_cores(const AllGatherParams& args, const Tensor& input_tensor) {
    auto* mesh_device = input_tensor.device();
    auto cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0)));
    if (args.sub_core_grid.has_value()) {
        cores = cores.intersection(args.sub_core_grid.value());
    }
    return cores;
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
    std::array<bool, 2> axis_is_straight{true, true};
    std::array<uint32_t, 2> axis_num_devices{1u, 1u};
    std::array<uint32_t, 2> axis_num_links{0u, 0u};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        const bool is_axis_active = mesh_shape[axis] > 1 && cluster_axis.value_or(axis) == axis;
        if (!is_axis_active) {
            continue;
        }
        axis_topology[axis] = ::ttnn::ccl::get_axis_topology(input_tensor, fabric_config, axis);
        axis_is_straight[axis] = ::ttnn::ccl::is_axis_straight(*mesh_device, axis);
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

    // Resolve the user's gather dim to a from-the-end index (see AllGatherParams::dim_from_end)
    const auto& logical_shape = input_tensor.logical_shape();
    const int32_t gather_dim_from_end =
        static_cast<int32_t>(logical_shape.get_normalized_index(dim)) - static_cast<int32_t>(logical_shape.rank());

    return {
        AllGatherParams{
            gather_dim_from_end,
            compute_output_specs_helper(input_tensor, gather_dim_from_end, num_devices, memory_config),
            cluster_axis,
            fabric_config,
            axis_topology,
            axis_is_straight,
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

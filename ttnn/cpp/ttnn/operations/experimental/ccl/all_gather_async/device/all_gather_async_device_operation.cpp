// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_device_operation.hpp"
#include "all_gather_async_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"  // for roofline calculation
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental::prim {

namespace {

bool via_broadcast_has_contiguous_output_slice(const Tensor& input_tensor, int32_t gather_dim) {
    const auto& padded_shape = input_tensor.padded_shape();
    std::vector<uint32_t> page_extents(padded_shape.begin(), padded_shape.end());
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(page_extents.size() >= 2, "Broadcast all-gather requires rank >= 2 for tile layout");
        page_extents[page_extents.size() - 2] =
            tt::div_up(page_extents[page_extents.size() - 2], tt::constants::TILE_HEIGHT);
        page_extents[page_extents.size() - 1] =
            tt::div_up(page_extents[page_extents.size() - 1], tt::constants::TILE_WIDTH);
    }

    return std::all_of(
        page_extents.begin(), page_extents.begin() + gather_dim, [](const auto& extent) { return extent == 1; });
}

}  // namespace

AllGatherAsyncVersion select_version(const AllGatherAsyncParams& operation_attributes) {
    // Check for minimal sharded case
    if (operation_attributes.use_all_gather_async_llama_sharded) {
        TT_FATAL(
            !operation_attributes.reverse_order,
            "Reversed all-gather (reverse_order=true) is not yet supported with llama-optimized variants "
            "(use_all_gather_async_llama_sharded=true). Please use the regular all_gather_async API instead of "
            "all_gather_async_reversed.");
        return AllGatherAsyncVersion::LLAMA_MINIMAL_SHARDED;
    }
    if (operation_attributes.use_all_gather_async_via_broadcast) {
        return AllGatherAsyncVersion::VIA_BROADCAST;
    }
    TT_FATAL(operation_attributes.semaphore.size() == 2, "Default implementation requires 2 semaphores");
    return AllGatherAsyncVersion::MINIMAL_DEFAULT;
}

AllGatherAsyncDeviceOperation::program_factory_t AllGatherAsyncDeviceOperation::select_program_factory(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& /*tensor_args*/) {
    AllGatherAsyncVersion version = select_version(args);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    switch (version) {
        case AllGatherAsyncVersion::LLAMA_MINIMAL_SHARDED: {
            return LlamaShardedMeshWorkloadFactory{};
        }
        case AllGatherAsyncVersion::VIA_BROADCAST: {
            return AllGatherViaBroadcastFactory{};
        }
        case AllGatherAsyncVersion::MINIMAL_DEFAULT:
        default: {
            return DefaultMeshWorkloadFactory{};
        }
    }
}

void AllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& layout = input_tensor.layout();
    const auto& dtype = input_tensor.dtype();
    const auto& page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") ||
            (input_tensor.memory_config().buffer_type() != BufferType::DRAM) ||
            !args.use_all_gather_async_llama_sharded,
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
    TT_FATAL(
        args.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    AllGatherAsyncVersion version = select_version(args);

    if (tensor_args.persistent_output_buffer.has_value()) {
        const auto& output_tensor = tensor_args.persistent_output_buffer.value();

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
        TT_FATAL(
            output_tensor.layout() == layout,
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.layout());
        TT_FATAL(
            output_tensor.dtype() == dtype,
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.tensor_spec().page_config());
        TT_FATAL(
            output_tensor.memory_config() == args.output_mem_config,
            "Error, Output tensor memory config should be same as output_mem_config but has {}",
            output_tensor.memory_config());

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.memory_config().memory_layout());

        // check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == args.dim) {
                TT_FATAL(
                    output_shape[i] <= input_shape[i] * args.ring_size,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] * args.ring_size,
                    output_shape[i]);
            } else {
                TT_FATAL(
                    output_shape[i] == input_shape[i],
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    output_shape[i]);
            }
        }

        if (version == AllGatherAsyncVersion::MINIMAL_DEFAULT) {
            // Checks specific to the MINIMAL_DEFAULT case

            // Don't support output DRAM block sharding
            if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(
                    output_tensor.memory_config().buffer_type() == BufferType::L1,
                    "We don't support output DRAM block sharding");
            }
        } else {
            // Checks specific to cases that are not MINIMAL_DEFAULT

            TT_FATAL(
                output_tensor.memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
                "Error, Output tensor memory layout should be same as input tensor memory layout but has {}",
                output_tensor.memory_config().memory_layout());
        }
    }

    // Checks specific to the MINIMAL_DEFAULT case
    if (version == AllGatherAsyncVersion::MINIMAL_DEFAULT) {
        // Don't support input DRAM block sharding
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                input_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support input DRAM block sharding");
        }
        TT_FATAL(input_tensor.logical_shape().rank() >= 2, "AllGatherAsync requires tensor of rank 2 or greater");
    } else {
        TT_FATAL(input_tensor.logical_shape().rank() == 4, "Llama specific all_gather requires tensor of rank 4");
    }

    if (version == AllGatherAsyncVersion::VIA_BROADCAST) {
        TT_FATAL(
            via_broadcast_has_contiguous_output_slice(input_tensor, args.dim),
            "Broadcast all-gather currently only supports gather dims whose preceding page-ordered dimensions are "
            "singleton. Got dim {} with padded shape {} and layout {}",
            args.dim,
            input_tensor.padded_shape(),
            input_tensor.layout());
    }
}

AllGatherAsyncDeviceOperation::spec_return_value_t AllGatherAsyncDeviceOperation::compute_output_specs(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] *= args.ring_size;
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

AllGatherAsyncDeviceOperation::tensor_return_value_t AllGatherAsyncDeviceOperation::create_output_tensors(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value() && args.using_persistent_buffers) {
        return tensor_args.persistent_output_buffer.value();
    }
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

AllGatherAsyncDeviceOperation::topology_return_value_t AllGatherAsyncDeviceOperation::compute_output_topologies(
    const AllGatherAsyncParams& args, const AllGatherAsyncInputs& tensor_args) {
    // After all_gather, the gathered axis is fully replicated across all devices on that axis.
    // Replace the placement on `cluster_axis` with Replicate, leaving other mesh axes untouched.
    // When cluster_axis is not provided, the op acts across the entire (1D) mesh, so all
    // placements are set to Replicate.
    const auto& input_topology = tensor_args.input_tensor.tensor_topology();
    auto output_placements = input_topology.placements();

    if (args.cluster_axis.has_value()) {
        const auto axis = args.cluster_axis.value();
        if (axis < output_placements.size()) {
            output_placements[axis] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    } else {
        for (auto& placement : output_placements) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    }

    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(output_placements), input_topology.mesh_coords())};
}

std::tuple<AllGatherAsyncParams, AllGatherAsyncInputs> all_gather_async_build_operation_args(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<uint32_t>& cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    bool use_all_gather_async_via_broadcast,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const MeshDevice* optional_mesh_device) {
    // Combine 3 implementations of the old all_gather_async_op.cpp::all_gather_async_impl
    // 1. only input_tensor, no output or optional mesh device
    // 2. has input tensor and output tensor but not optional mesh device
    // 3. has all three

    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    bool using_persistent_buffers = persistent_output_buffer.has_value();

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    // Prioritize optional mesh device first, then check device of input_tensor
    bool using_optional_mesh_device = optional_mesh_device != nullptr;
    if (using_optional_mesh_device) {
        const auto& mesh_view = optional_mesh_device->get_view();
        TT_FATAL(
            mesh_view.is_mesh_2d(),
            "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    } else {
        TT_FATAL(input_tensor.device() != nullptr, "Input tensor has no mesh device assigned.");

        TT_FATAL(num_devices > 1, "all_gather_async op will only work for num_devices > 1, but has {}", num_devices);

        if (using_persistent_buffers) {
            log_debug(tt::LogOp, "creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
            log_debug(tt::LogOp, "line_fabric is created");
        }
    }

    return {
        AllGatherAsyncParams(
            gather_dim,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis,
            use_all_gather_async_llama_sharded,
            use_optimal_ccl_for_llama,
            use_all_gather_async_via_broadcast,
            barrier_semaphore,
            using_persistent_buffers,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            reverse_order,
            sub_core_grid),
        AllGatherAsyncInputs{.input_tensor = input_tensor, .persistent_output_buffer = persistent_output_buffer}};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<AllGatherAsyncDeviceOperation::tensor_return_value_t>
AllGatherAsyncDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    // =========================================================================
    // AllGather Roofline Performance Model (First-Principles)
    //
    // AllGather is pure data movement — no compute. The roofline is the
    // hardware ceiling: the minimum time given the physical topology and
    // memory bandwidth, independent of any particular algorithm.
    //
    // Performance is the slowest single resource, not the sum: a pipelined
    // collective overlaps fill/drain with streaming, so the terms compete (max):
    //   ideal_cycles = max(DRAM_bw, fabric_bw, DRAM_fill/drain, fabric_fill)
    //
    // --- Fabric term (bottleneck link analysis) ---
    //
    // For any topology the roofline is set by the most-loaded link.
    // N devices, each contributing S bytes. Every device must receive
    // the other (N-1) chunks.
    //
    // LINE topology:
    //   The edge device has a single link to the rest of the network.
    //   All (N-1) chunks must pass through that one link — no topology
    //   can avoid this.
    //   bottleneck_bytes = (N-1) * S.  Max hops = N-1.
    //
    // RING topology:
    //   A bisection cut crosses 2 links. Data from N/2 devices on one
    //   side must reach N/2 devices on the other, and vice-versa. Each
    //   direction carries at most half the total load.
    //   bottleneck_bytes = ceil((N-1) * S / 2).  Max hops = ceil((N-1)/2).
    //
    // --- DRAM term (memory bandwidth ceiling) ---
    //
    // Each device reads S bytes and writes N*S bytes. The roofline
    // assumes all compute cores can drive DRAM concurrently (hardware
    // maximum parallelism), so DRAM time = bytes / peak_aggregate_BW.
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

    const uint32_t N = args.ring_size;
    const uint32_t num_links = args.num_links;
    uint64_t bottleneck_bytes = 0;  // bottleneck bytes through the most-loaded link
    uint32_t num_hops = 0;          // collective diameter (hops)
    if (N <= 1) {
        // Single device: no fabric communication
    } else if (tt::tt_fabric::is_ring_or_torus(args.topology)) {
        // Ring topology: bisection cuts 2 links, so each direction carries
        // at most half the total data. Bottleneck per direction = ceil((N-1)*S/2).
        bottleneck_bytes = tt::div_up((N - 1) * input_size_bytes, 2);
        num_hops = tt::div_up(N - 1, 2u);
    } else {
        // Line/Linear/Mesh topology: edge device has one link and must
        // receive all (N-1) slices through it.
        bottleneck_bytes = (N - 1) * input_size_bytes;
        num_hops = N - 1;
    }

    // Fabric gives two floors: bandwidth (bottleneck-link bytes / BW) and fill latency
    // (first chunk crossing the diameter). Fill overlaps streaming rather than preceding
    // it, so it joins the max() below instead of adding on.
    const auto [fabric_bw_cycles, fabric_fill_cycles] = ttnn::ccl::estimate_fabric_transfer_cycles(
        arch, tt::tt_fabric::GetFabricConfig(), clock_rate_mhz, bottleneck_bytes, num_links, num_hops);

    // --- Local DRAM bandwidth ceiling (first-principles) ---
    // Read: device reads S bytes from DRAM.  Write: device writes N*S bytes.
    // Roofline assumes all device compute cores drive DRAM concurrently
    // (hardware max parallelism). DRAM bandwidth and its read/write fill-drain
    // latency are two more floors; all of these overlap, so all feed the final max().
    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const uint32_t read_page_size = input_tensor.buffer()->page_size();
    const uint32_t write_page_size = output_tensor.buffer()->page_size();

    // Hardware ceiling: total compute cores available to drive DRAM
    const uint32_t device_rows = input_tensor.device()->compute_with_storage_grid_size().x;
    const uint32_t device_cols = input_tensor.device()->compute_with_storage_grid_size().y;
    const uint32_t num_cores = device_rows * device_cols;

    const int64_t output_size_bytes = N * input_size_bytes;
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
    // Optimistic floor = the slowest single resource, not the sum. A pipelined collective overlaps
    // fill/drain with streaming (nearer chunks arrive while the farthest is still in flight): the
    // throughput ceiling and the fill/drain floor each take a max, then those two compete (not +).
    const int throughput_cycles = std::max(local_bw_cycles, fabric_bw_cycles);
    const int fill_cycles = std::max(pipeline_latency_cycles, fabric_fill_cycles);
    const int ideal_dev_clock_cycles = std::max(throughput_cycles, fill_cycles);

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor all_gather_async(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<uint32_t>& cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    bool use_all_gather_async_via_broadcast,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const MeshDevice* optional_mesh_device) {
    auto [params, inputs] = experimental::prim::all_gather_async_build_operation_args(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        sub_device_id,
        cluster_axis,
        use_optimal_ccl_for_llama,
        use_all_gather_async_llama_sharded,
        use_all_gather_async_via_broadcast,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        reverse_order,
        sub_core_grid,
        optional_mesh_device);
    return ttnn::device_operation::launch<experimental::prim::AllGatherAsyncDeviceOperation>(params, inputs);
}

}  // namespace ttnn::prim

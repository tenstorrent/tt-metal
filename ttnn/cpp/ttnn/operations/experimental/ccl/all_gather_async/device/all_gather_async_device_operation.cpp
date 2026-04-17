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
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

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

ttsl::hash::hash_t AllGatherAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    if (args.sub_core_grid.has_value()) {
        subdevice_core_range_set = subdevice_core_range_set.intersection(args.sub_core_grid.value());
    }

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllGatherAsyncDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_mem_config,
        args.topology,
        args.cluster_axis,
        args.barrier_semaphore.has_value(),
        args.using_persistent_buffers,
        args.chunks_per_sync,
        args.num_workers_per_link,
        args.num_buffers_per_channel,
        args.use_all_gather_async_llama_sharded,
        args.use_optimal_ccl_for_llama,
        args.reverse_order,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
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
    // AllGather Roofline Performance Model
    //
    // AllGather is pure data movement — no compute. Performance is bounded by:
    //   ideal_ns = max(DRAM_bandwidth_time, Fabric_transfer_time)
    //
    // The OpPerformanceModelGeneral constructor computes DRAM_bandwidth_time
    // from input/output tensor sizes and peak DRAM BW. We encode the fabric
    // transfer time as ideal_compute_cycles so the constructor's max() picks
    // the true bottleneck.
    //
    // IMPORTANT: All N devices multicast simultaneously, so fabric links are
    // shared. The bottleneck link is the most congested one — it carries
    // traffic from multiple senders. We model the total bytes through that
    // bottleneck link, not just one device's contribution.
    //
    // Ideal algorithms by topology (conceptual, not kernel-specific):
    //
    // LINE topology:
    //   Each device multicasts S bytes to all others. Both directions run in
    //   parallel. The bottleneck link (near the center) carries traffic from
    //   ceil(N/2) senders, so bottleneck_bytes = ceil(N/2) * S.
    //   Latency = (N-1) hops for the farthest slice.
    //
    // RING topology — pick the better of two strategies:
    //   Option A (low latency, high BW):
    //     Each device multicasts full slice S to its half-ring (ceil(N/2) hops).
    //     The bottleneck link carries traffic from ceil(N/2) senders.
    //     bottleneck_bytes = ceil(N/2) * S, latency = ceil(N/2) hops.
    //
    //   Option B (low BW, high latency):
    //     Each device splits its slice in half and multicasts S/2 in each
    //     direction around the full ring (N-1 hops). The bottleneck link
    //     carries traffic from all N senders (each sending S/2 one way).
    //     bottleneck_bytes = N * S/2, latency = (N-1) hops.
    //
    //   fabric_ns = min(time_A, time_B)
    // =========================================================================

    const auto& input_tensor = tensor_args.input_tensor;

    // --- Architecture and clock detection ---
    const auto arch =
        input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : tt::ARCH::WORMHOLE_B0;

    float clock_rate_ghz;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        int freq_mhz = input_tensor.device()->get_clock_rate_mhz();
        clock_rate_ghz = freq_mhz / 1000.0f;
    } else {
        clock_rate_ghz = 1.0f;
    }

    // --- Data size: bytes each device contributes ---
    const int64_t input_size_bytes =
        static_cast<int64_t>(input_tensor.physical_volume()) * static_cast<int64_t>(input_tensor.element_size());

    // --- Assumed packet size for BW map lookup (easy to edit) ---
    const uint32_t packet_size = 4096;

    const uint32_t N = args.ring_size;
    const uint32_t num_links = args.num_links;
    const uint32_t half_N = (N + 1) / 2;  // ceil(N/2)

    float fabric_time_ns = 0.0f;

    if (N <= 1) {
        // Single device: no fabric communication
        fabric_time_ns = 0.0f;
    } else if (tt::tt_fabric::is_ring_or_torus(args.topology)) {
        // Ring topology: pick the better of two strategies

        // Option A: multicast full slice to each half-ring.
        // Bottleneck link carries ceil(N/2) senders * S bytes each.
        const int64_t bottleneck_bytes_a = static_cast<int64_t>(half_N) * input_size_bytes;
        const float time_a = ttnn::ccl::estimate_fabric_transfer_ns(
            bottleneck_bytes_a, num_links, packet_size, /*is_multicast=*/true, half_N, arch);

        // Option B: split slice, multicast S/2 in each direction around full ring.
        // Bottleneck link carries all N senders * S/2 bytes each.
        const int64_t bottleneck_bytes_b = static_cast<int64_t>(N) * (input_size_bytes / 2);
        const float time_b = ttnn::ccl::estimate_fabric_transfer_ns(
            bottleneck_bytes_b, num_links, packet_size, /*is_multicast=*/true, N - 1, arch);

        fabric_time_ns = std::min(time_a, time_b);
    } else {
        // Line/Linear/Mesh topology: each device multicasts S to all others.
        // Bottleneck link (near center) carries ceil(N/2) senders * S bytes.
        const int64_t bottleneck_bytes = static_cast<int64_t>(half_N) * input_size_bytes;
        fabric_time_ns = ttnn::ccl::estimate_fabric_transfer_ns(
            bottleneck_bytes, num_links, packet_size, /*is_multicast=*/true, N - 1, arch);
    }

    // Convert fabric time (ns) to device clock cycles.
    // clock_rate_ghz cycles/ns * ns = cycles
    const int ideal_dev_clock_cycles = static_cast<int>(std::ceil(fabric_time_ns * clock_rate_ghz));

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

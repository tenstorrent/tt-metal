// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ReduceScatterMinimalAsyncDeviceOperation::program_factory_t
ReduceScatterMinimalAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
        return RingReduceScatterMeshWorkloadFactory{};
    }
    TT_FATAL(operation_attributes.topology == ttnn::ccl::Topology::Linear, "Topology must be Ring or Linear");
    return LineReduceScatterMeshWorkloadFactory{};
}

void ReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Lightweight validation for cache hits
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void ReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Common validation
    ttnn::experimental::ccl::reduce_scatter_common_validates(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        tensor_args.optional_output_tensor);

    // Validate intermediate tensor if provided
    if (tensor_args.optional_intermediate_tensor.has_value()) {
        ttnn::experimental::ccl::validate_intermediate_tensor(
            input_tensor,
            tensor_args.optional_intermediate_tensor.value(),
            operation_attributes.optional_intermediate_mem_config);
    }

    // Validate semaphore count
    constexpr auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Expected {} semaphores but got {}",
        num_expected_semaphores,
        operation_attributes.semaphore.size());
}

std::vector<ttnn::TensorSpec> ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto inter_shape = input_tensor.padded_shape();

    MemoryConfig adjusted_intermediate_mem_config,
        intermediate_mem_config =
            operation_attributes.optional_intermediate_mem_config.value_or(input_tensor.memory_config());

    if (operation_attributes.topology == ttnn::ccl::Topology::Linear) {
        inter_shape[0] *= 2;

        // Adjust memory config for sharded tensors
        if (intermediate_mem_config.is_sharded() &&
            !operation_attributes.optional_intermediate_mem_config.has_value()) {
            auto intermediate_shard_spec = intermediate_mem_config.shard_spec().value();
            intermediate_shard_spec.shape[0] *= 2;
            adjusted_intermediate_mem_config = intermediate_mem_config.with_shard_spec(intermediate_shard_spec);
        } else {
            adjusted_intermediate_mem_config = intermediate_mem_config;
        }
    } else {
        adjusted_intermediate_mem_config = intermediate_mem_config;
    }

    auto output_shape = input_tensor.logical_shape();
    output_shape[operation_attributes.dim] /= operation_attributes.ring_size;

    return {
        TensorSpec(
            inter_shape,
            TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), adjusted_intermediate_mem_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                operation_attributes.output_mem_config)),
    };
}

std::vector<Tensor> ReduceScatterMinimalAsyncDeviceOperation::create_output_tensors(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    auto tensor_specs = compute_output_specs(operation_attributes, tensor_args);
    const auto& input_tensor = tensor_args.input_tensor;

    ttnn::Tensor intermediate_buffer = tensor_args.optional_intermediate_tensor.has_value()
                                           ? tensor_args.optional_intermediate_tensor.value()
                                           : create_device_tensor(tensor_specs[0], input_tensor.device());

    ttnn::Tensor output_buffer = tensor_args.optional_output_tensor.has_value()
                                     ? tensor_args.optional_output_tensor.value()
                                     : create_device_tensor(tensor_specs[1], input_tensor.device());

    return {intermediate_buffer, output_buffer};
}

ttsl::hash::hash_t ReduceScatterMinimalAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "ReduceScatterMinimalAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<ReduceScatterMinimalAsyncDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.optional_intermediate_mem_config,
        operation_attributes.topology,
        operation_attributes.barrier_semaphore.has_value(),
        operation_attributes.using_persistent_buffers,
        operation_attributes.cluster_axis,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        operation_attributes.compute_kernel_config,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ReduceScatterMinimalAsyncDeviceOperation::tensor_return_value_t>
ReduceScatterMinimalAsyncDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    // =========================================================================
    // ReduceScatter Roofline Performance Model
    //
    // ReduceScatter: N devices, each has S bytes of input. Each device's
    // output is the element-wise reduction (sum) of its chunk across all
    // devices: output size = S/N bytes per device.
    //
    // Unlike AllGather where data is forwarded unmodified (each link carries
    // full S-byte slices), ReduceScatter REDUCES data at each hop — each
    // link only carries S/N-byte chunks. This makes ReduceScatter N× more
    // communication-efficient per link than AllGather.
    //
    // The model uses bottleneck analysis over total resource utilization.
    // All resources operate in parallel for the pipeline duration T.
    // T must be long enough for every resource to finish its total work:
    //
    //   T >= work_i / BW_i   for each resource i
    //   T  = max(all lower bounds)
    //
    // Different resources have different total data volumes (the N-dependence
    // varies per resource), so you cannot factor out N. The max() over
    // total work/BW correctly identifies the binding constraint.
    //
    // Double buffering: while a device processes the current slice
    // (read + reduce + send), it receives the next slice in parallel.
    // This overlap is captured by the max() — the receive and process
    // sides compete only for shared bandwidth (e.g. DRAM).
    // =========================================================================

    const auto& input_tensor = tensor_args.input_tensor;

    // --- Architecture and clock ---
    tt::ARCH arch;
    float clock_rate_ghz;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        arch = input_tensor.device()->arch();
        clock_rate_ghz = input_tensor.device()->get_clock_rate_mhz() / 1000.0f;
    } else {
        arch = tt::ARCH::WORMHOLE_B0;
        clock_rate_ghz = 1.0f;
    }

    // --- Data sizes ---
    const uint64_t input_size_bytes =
        static_cast<uint64_t>(input_tensor.physical_volume()) * input_tensor.element_size();
    const uint32_t N = args.ring_size;
    const uint32_t num_links = args.num_links;
    const uint64_t chunk_size = input_size_bytes / N;  // S/N bytes per device output

    // =========================================================================
    // 1. FABRIC TRANSFER
    //
    // We model from the worst-case device's perspective: the one whose
    // busiest incoming link carries the most data.
    //
    // RING topology — half-ring algorithm:
    //   For chunk c_i (destined for device i), the other N-1 devices are
    //   split into two halves: clockwise and counter-clockwise of device i.
    //   Each half pipelines partial reductions toward i from its direction.
    //   Device i does a final local reduction of the two arriving partials.
    //
    //   Through any link in one direction, exactly floor(N/2) chunks of
    //   S/N bytes pass (verified by enumeration for even and odd N).
    //     Bottleneck bytes per direction = floor(N/2) × S/N
    //     Latency = floor(N/2) hops (farthest partial travels half the ring)
    //
    //   Half-ring strictly dominates full-ring (unidirectional) for all N:
    //     Full-ring: (N-1)×S/N bytes, N-1 hops — worse on both BW and latency.
    //
    // LINE topology — edge device bottleneck:
    //   Edge device (D0 or DN-1) has one link. It must send its (N-1)
    //   chunk contributions outward through that single link.
    //     Bottleneck bytes = (N-1) × S/N
    //     Latency = N-1 hops
    // =========================================================================
    double fabric_time_ns = 0.0;
    if (N <= 1) {
        fabric_time_ns = 0.0;
    } else if (args.topology == ttnn::ccl::Topology::Ring) {
        const uint64_t bottleneck_bytes = static_cast<uint64_t>(N / 2) * chunk_size;
        const uint32_t num_hops = N / 2;
        fabric_time_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, bottleneck_bytes, num_links, num_hops);
    } else {
        // Line/Linear topology
        const uint64_t bottleneck_bytes = static_cast<uint64_t>(N - 1) * chunk_size;
        const uint32_t num_hops = N - 1;
        fabric_time_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, bottleneck_bytes, num_links, num_hops);
    }
    const int fabric_cycles = static_cast<int>(std::ceil(fabric_time_ns * clock_rate_ghz));

    // =========================================================================
    // 2. LOCAL DATA MOVEMENT (DRAM/L1 read and write)
    //
    // Total DRAM read per device:
    //   - Input tensor read: S bytes. The device reads its full input because
    //     it contributes its own S/N-byte chunk to every reduction that passes
    //     through (N chunks total, one per pipeline step).
    //   - Intermediate buffer read: (N-1) × S/N bytes. Each chunk received
    //     from fabric is written to the intermediate buffer, then read back
    //     for reduction processing.
    //   Total read = S + (N-1)×S/N = S×(2N-1)/N
    //
    // Total DRAM write per device:
    //   At each pipeline step, two things happen in parallel:
    //     Receive side: next slice arrives from fabric → written to intermediate
    //     Process side: current slice is read from intermediate + own input,
    //       reduced in L1, then forwarded via fabric (L1 → fabric, no DRAM write)
    //   Only the final step writes the reduced output to DRAM.
    //
    //   - Intermediate buffer write: (N-1) × S/N bytes (fabric → DRAM, receive side)
    //   - Output write: S/N bytes (final reduced chunk → DRAM, last step only)
    //   Total write = (N-1)×S/N + S/N = N×S/N = S
    //
    // DRAM read and write can overlap (different NOC channels), so we take
    // max(read_bw, write_bw) rather than summing them.
    //
    // Note: intermediate buffer generally uses the same memory type as
    // input (DRAM for DRAM inputs). We use input's buffer type for reads
    // and output's buffer type for writes.
    // =========================================================================
    const auto& output = output_tensors[1];  // [0] is intermediate, [1] is output
    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output.buffer()->buffer_type() == BufferType::DRAM;
    const uint32_t page_size = input_tensor.buffer()->page_size();

    const uint64_t total_read_bytes = input_size_bytes + static_cast<uint64_t>(N - 1) * chunk_size;
    const uint32_t read_pages_per_worker = tt::div_up(static_cast<uint32_t>(total_read_bytes / page_size), num_links);

    const uint64_t total_write_bytes = static_cast<uint64_t>(N - 1) * chunk_size + chunk_size;
    const uint32_t write_pages_per_worker = tt::div_up(static_cast<uint32_t>(total_write_bytes / page_size), num_links);

    auto [read_bw_cycles, read_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        page_size, input_is_dram, /*is_local=*/false, read_pages_per_worker, arch, /*is_read=*/true);
    auto [write_bw_cycles, write_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        page_size, output_is_dram, /*is_local=*/false, write_pages_per_worker, arch, /*is_read=*/false);

    // =========================================================================
    // 3. COMPUTE (element-wise reduction)
    //
    // Each device performs (N-1) element-wise reductions (add_tiles).
    // Each reduction takes two S/N-byte inputs and produces one S/N-byte output.
    //
    // The bottleneck is the unpacker that feeds tiles from L1 circular
    // buffers to the math engine. Each Tensix core has two unpackers
    // (one for SrcA, one for SrcB) sharing L1 access ports. Each can
    // run at up to x4 speed (64 B/cycle), but their joint maximum is
    // five 128-bit reads per cycle = 80 B/cycle. This limit applies
    // to all architectures (Grayskull, Wormhole B0, Blackhole).
    //
    // The math engine itself is faster — 128 datums/cycle for element-wise
    // add (8×16 FPU at LoFi) — so the unpacker is the binding constraint.
    //
    // Per reduction: unpack 2 inputs = 2 × S/N bytes through the unpacker.
    // Total: 2 × (N-1) × S/N bytes across all reductions.
    // Distributed across num_links worker cores.
    //
    // compute_cycles_per_worker = 2×(N-1) × (S/N) / num_links / 80
    // =========================================================================
    constexpr uint32_t UNPACKER_BW_BYTES_PER_CYCLE = 80;
    const uint64_t total_unpack_bytes = 2ULL * (N - 1) * chunk_size;
    const int compute_cycles =
        static_cast<int>(total_unpack_bytes / (static_cast<uint64_t>(num_links) * UNPACKER_BW_BYTES_PER_CYCLE));

    // =========================================================================
    // 4. PIPELINED MODEL
    //
    // All resources operate in parallel throughout the pipeline.
    // BW terms compete — the slowest resource determines the throughput:
    //   max(fabric, dram_read_bw, dram_write_bw, compute)
    // Latencies are additive — they represent pipeline fill (first read)
    // and drain (last write) that cannot overlap with steady-state.
    // =========================================================================
    const int local_bw_cycles = static_cast<int>(std::max(read_bw_cycles, write_bw_cycles));
    const int pipeline_latency_cycles = static_cast<int>(read_latency_cycles + write_latency_cycles);
    const int ideal_dev_clock_cycles =
        std::max(std::max(local_bw_cycles, fabric_cycles), compute_cycles) + pipeline_latency_cycles;

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, output_tensors, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> reduce_scatter_minimal_async(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    MemoryConfig output_mem_config,
    std::optional<MemoryConfig> optional_intermediate_mem_config,
    ttnn::ccl::Topology topology,
    std::vector<GlobalSemaphore> semaphore,
    std::optional<GlobalSemaphore> barrier_semaphore,
    bool using_persistent_buffers,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::ReduceScatterMinimalAsyncDeviceOperation;
    const auto resolved_sub_device_id = sub_device_id.value_or(input_tensor.device()->get_sub_device_ids().at(0));

    auto operation_attributes = OperationType::operation_attributes_t{
        dim,
        num_links,
        ring_size,
        std::move(output_mem_config),
        std::move(optional_intermediate_mem_config),
        topology,
        std::move(semaphore),
        std::move(barrier_semaphore),
        using_persistent_buffers,
        resolved_sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, optional_intermediate_tensor, optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

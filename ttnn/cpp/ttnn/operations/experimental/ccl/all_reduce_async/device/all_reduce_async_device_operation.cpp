// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_device_operation.hpp"
#include "all_reduce_async_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"  // for roofline calculation

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {
void AllReduceAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;
    const auto& page_size = input_tensor.buffer()->page_size();

    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") ||
            (input_tensor.memory_config().buffer_type() != BufferType::DRAM),
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllReduceAsync currently requires aligned pages");
    TT_FATAL(
        args.ring_size % 2 == 0,
        "AllReduceAsync currently only supports even number of blocks in the reduction kernel.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(buffer_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(buffer_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
    TT_FATAL(
        args.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for input tensor{}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for buffer tensor {}.",
        buffer_tensor.memory_config().memory_layout());
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for output tensor {}.",
        args.output_mem_config.memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().shard_spec()->grid.contains(args.output_mem_config.shard_spec()->grid),
        "The output tensor must reside on a subset of the cores of the buffer tensor");

    const uint32_t output_shard_shape_volume =
        args.output_mem_config.shard_spec()->shape[0] * args.output_mem_config.shard_spec()->shape[1];
    const uint32_t buffer_shard_shape_volume =
        buffer_tensor.memory_config().shard_spec()->shape[0] * buffer_tensor.memory_config().shard_spec()->shape[1];
    TT_FATAL(
        output_shard_shape_volume * args.ring_size <= buffer_shard_shape_volume,
        "The shard size for the buffer must be large enough to hold the intermediate tensor. Require at least {} but "
        "has {}",
        output_shard_shape_volume * args.ring_size,
        buffer_shard_shape_volume);
}

AllReduceAsyncDeviceOperation::spec_return_value_t AllReduceAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    tt::tt_metal::TensorLayout output_tensor_layout =
        tt::tt_metal::TensorLayout(args.dtype, input_tensor.tensor_spec().page_config(), args.output_mem_config);

    return TensorSpec(shape, output_tensor_layout);
}

AllReduceAsyncDeviceOperation::tensor_return_value_t AllReduceAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

ttsl::hash::hash_t AllReduceAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllReduceAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<AllReduceAsyncDeviceOperation>(
        args.num_links,
        args.ring_size,
        args.dtype,
        args.output_mem_config,
        args.topology,
        args.use_noc1_only,
        args.use_optimal_ccl_for_llama,
        args.cluster_axis,
        subdevice_core_range_set,
        tensor_args);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<AllReduceAsyncDeviceOperation::tensor_return_value_t>
AllReduceAsyncDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    // =========================================================================
    // AllReduce Roofline Performance Model (First-Principles)
    //
    // AllReduce: N devices, each has S bytes. After the operation every
    // device holds the element-wise reduction of all inputs (output = S bytes).
    //
    // Decomposition: AllReduce = ReduceScatter + AllGather.
    // This is bandwidth-optimal and lets us reason about the minimum data
    // that must traverse each resource, independent of any algorithm.
    //
    // The two phases share the same physical links sequentially, so their
    // fabric data volumes are additive on the bottleneck link. DRAM and
    // compute are modeled at the AllReduce level (input→output), not by
    // summing RS and AG independently.
    //
    // Performance is bounded by:
    //   ideal_cycles = max(fabric_cycles, dram_bw_cycles, compute_cycles)
    //                  + pipeline_latency
    // =========================================================================

    const auto& input_tensor = tensor_args.input_tensor;

    // --- Architecture and clock ---
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    float clock_rate_ghz = 1.0f;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        arch = input_tensor.device()->arch();
        clock_rate_ghz = input_tensor.device()->get_clock_rate_mhz() / 1000.0f;
    }

    // --- Data sizes ---
    const uint64_t S = static_cast<uint64_t>(input_tensor.physical_volume()) * input_tensor.element_size();
    const uint32_t N = args.ring_size;
    const uint32_t num_links = args.num_links;
    const uint64_t slice_size = S / N;  // S/N bytes — per-device reduced chunk

    // =========================================================================
    // 1. FABRIC TRANSFER (algorithm-agnostic lower bound)
    //
    // AllReduce = ReduceScatter + AllGather on the same physical links.
    // Both phases contribute bandwidth time and pipeline-fill latency,
    // so we call estimate_fabric_transfer_ns separately and sum.
    //
    // RING topology (bisection argument):
    //   RS phase — bisection link carries (N-1)*S/(2N) bytes.
    //   AG phase — bisection link carries ceil((N-1)*S/2) bytes.
    //   Hops per phase: ceil((N-1)/2) (ring diameter).
    //
    // LINEAR topology (edge-device bottleneck):
    //   RS phase — edge link carries (N-1)*S/N bytes.
    //   AG phase — edge link carries (N-1)*S bytes.
    //   Hops per phase: N-1 (linear diameter).
    // =========================================================================
    double fabric_time_ns = 0.0;
    if (N <= 1) {
        fabric_time_ns = 0.0;
    } else if (tt::tt_fabric::is_ring_or_torus(args.topology)) {
        const uint64_t rs_bottleneck_bytes = tt::div_up((N - 1) * slice_size, 2);
        const uint64_t ag_bottleneck_bytes = tt::div_up((N - 1) * S, 2);
        const uint32_t num_hops = tt::div_up(N - 1, 2u);
        double rs_fabric_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, rs_bottleneck_bytes, num_links, num_hops);
        double ag_fabric_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, ag_bottleneck_bytes, num_links, num_hops);
        fabric_time_ns = rs_fabric_ns + ag_fabric_ns;
    } else {
        const uint64_t rs_bottleneck_bytes = (N - 1) * slice_size;
        const uint64_t ag_bottleneck_bytes = (N - 1) * S;
        const uint32_t num_hops = N - 1;
        double rs_fabric_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, rs_bottleneck_bytes, num_links, num_hops);
        double ag_fabric_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, ag_bottleneck_bytes, num_links, num_hops);
        fabric_time_ns = rs_fabric_ns + ag_fabric_ns;
    }
    const int fabric_cycles = static_cast<int>(std::ceil(fabric_time_ns * clock_rate_ghz));

    // =========================================================================
    // 2. LOCAL DATA MOVEMENT — first-principles minimum
    //
    // DRAM reads:  S bytes — input tensor read once.
    // DRAM writes: S bytes — output tensor (same shape as input, fully reduced).
    // The RS→AG intermediate (S/N bytes) stays in L1 in the optimal case;
    // roofline models the hardware ceiling.
    //
    // Read and write can overlap (different NOC channels), so we take
    // max(read_bw, write_bw) rather than summing.
    // =========================================================================
    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output_tensors.buffer()->buffer_type() == BufferType::DRAM;
    const uint32_t read_page_size = input_tensor.buffer()->page_size();
    const uint32_t write_page_size = output_tensors.buffer()->page_size();
    const uint32_t device_rows = input_tensor.device()->compute_with_storage_grid_size().x;
    const uint32_t device_cols = input_tensor.device()->compute_with_storage_grid_size().y;
    const uint32_t num_cores = device_rows * device_cols;

    const uint32_t read_pages = tt::div_up(S, read_page_size);
    const uint32_t write_pages = tt::div_up(S, write_page_size);
    const uint32_t read_pages_per_core = tt::div_up(read_pages, num_cores);
    const uint32_t write_pages_per_core = tt::div_up(write_pages, num_cores);

    auto [read_bw_cycles, read_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        read_page_size, input_is_dram, /*is_local=*/false, read_pages_per_core, arch, /*is_read=*/true);
    auto [write_bw_cycles, write_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        write_page_size, output_is_dram, /*is_local=*/false, write_pages_per_core, arch, /*is_read=*/false);

    // =========================================================================
    // 3. COMPUTE (element-wise reduction) — algorithm-agnostic
    //
    // (N-1) element-wise reductions, each on two S/N-byte inputs.
    // Total unpack bytes: 2 * (N-1) * S/N.
    // Bottleneck is the unpacker (80 B/cycle joint SrcA+SrcB limit),
    // distributed across all compute cores.
    // =========================================================================
    constexpr uint32_t UNPACKER_BW_BYTES_PER_CYCLE = 80;
    const uint64_t total_unpack_bytes = 2ULL * (N - 1) * slice_size;
    const int compute_cycles =
        tt::div_up(total_unpack_bytes, static_cast<uint64_t>(num_cores) * UNPACKER_BW_BYTES_PER_CYCLE);

    // =========================================================================
    // 4. PIPELINED MODEL
    //
    // BW terms compete — the slowest resource sets throughput:
    //   max(fabric, dram_read_bw, dram_write_bw, compute)
    //
    // Latencies are additive (pipeline fill/drain):
    //   read_latency  — first DRAM read before pipeline starts
    //   compute_latency — last chunk's reduction after all data arrived
    //   write_latency — last DRAM write after last reduction completes
    // =========================================================================
    const int local_bw_cycles = static_cast<int>(std::max(read_bw_cycles, write_bw_cycles));
    const int compute_latency_cycles =
        static_cast<int>(2ULL * slice_size / (static_cast<uint64_t>(num_cores) * UNPACKER_BW_BYTES_PER_CYCLE));
    const int pipeline_latency_cycles =
        static_cast<int>(read_latency_cycles) + compute_latency_cycles + static_cast<int>(write_latency_cycles);
    const int ideal_dev_clock_cycles =
        std::max({local_bw_cycles, fabric_cycles, compute_cycles}) + pipeline_latency_cycles;

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensors}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::AllReduceAsyncDeviceOperation::tensor_return_value_t all_reduce_async(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    uint32_t cluster_axis,
    MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    std::optional<DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    using OperationType = ttnn::experimental::prim::AllReduceAsyncDeviceOperation;
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-reduce invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    auto operation_attributes = OperationType::operation_attributes_t(
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        dtype.value_or(input_tensor.dtype()),
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        use_noc1_only,
        use_optimal_ccl_for_llama,
        cluster_axis,
        &mesh_device);
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor, .buffer_tensor = buffer_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

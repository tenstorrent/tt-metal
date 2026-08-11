// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"  // for roofline calculation
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

// True when this call stages through the chunk-paged contiguous intermediate, which additionally needs
// the penult intermediate as a third output tensor.
//
// compute_output_specs, create_output_tensors and compute_output_topologies must agree exactly on the
// output count (the framework TT_FATALs if the topology count and the tensor count differ), and the
// ring program factory keys off the same answer to decide whether it has a penult intermediate to wire up.
// Routing all four through this one predicate is what keeps them from disagreeing.
bool uses_contiguous_staging(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    return ttnn::experimental::ccl::reduce_scatter_use_contiguous_interm(
        tensor_args.input_tensor,
        tensor_args.optional_intermediate_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.ring_size,
        ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config));
}

}  // unnamed namespace

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

    if (tensor_args.optional_intermediate_tensor.has_value()) {
        TT_FATAL(
            tensor_args.optional_intermediate_tensor->device() == input_tensor.device(),
            "Persistent intermediate tensor must be allocated on the same mesh device as the input tensor");
    }
    if (tensor_args.optional_penult_intermediate_tensor.has_value()) {
        TT_FATAL(
            tensor_args.optional_penult_intermediate_tensor->device() == input_tensor.device(),
            "Persistent penult intermediate tensor must be allocated on the same mesh device as the input tensor");
    }
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

    // Validate intermediate tensor if provided.
    //
    // The ring path (scatter dim != 0) supports two intermediate layouts, and a caller-provided
    // buffer selects between them by matching one of the two specs:
    //   - the chunk-paged contiguous staging buffer (UINT8 row-major interleaved DRAM), or
    //   - an input-shaped tiled intermediate.
    // Anything matching neither is rejected here, so reduce_scatter_use_contiguous_interm can treat
    // "does it match the staging spec" as a decision downstream.
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    const auto stage_spec = ttnn::experimental::ccl::reduce_scatter_ring_interm_staging_spec(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.ring_size,
        fp32_dest_acc_en);
    const bool use_contiguous = uses_contiguous_staging(operation_attributes, tensor_args);

    if (tensor_args.optional_intermediate_tensor.has_value()) {
        const auto& interm = tensor_args.optional_intermediate_tensor.value();
        TT_FATAL(interm.storage_type() == StorageType::DEVICE, "Persistent intermediate tensor must be on device");
        TT_FATAL(
            interm.device() == input_tensor.device(),
            "Persistent intermediate tensor must be allocated on the same mesh device as the input tensor");
        if (!use_contiguous) {
            if (stage_spec.has_value()) {
                // The contiguous layout was available and this buffer is not it, so the caller must
                // be asking for the tiled path. Report both options if it is neither.
                TT_FATAL(
                    interm.padded_shape() == input_tensor.padded_shape() && interm.layout() == input_tensor.layout() &&
                        interm.dtype() == input_tensor.dtype(),
                    "Persistent intermediate matches neither reduce-scatter intermediate layout. Expected either "
                    "the contiguous chunk-paged staging buffer (shape {}, UINT8, row-major, interleaved DRAM; "
                    "allocate it with reduce_scatter_minimal_async_create_intermediate_buffer) or an input-shaped "
                    "tiled intermediate (padded shape {}, dtype {}, layout {}). Got padded shape {}, dtype {}, "
                    "layout {}.",
                    stage_spec->logical_shape(),
                    input_tensor.padded_shape(),
                    input_tensor.dtype(),
                    input_tensor.layout(),
                    interm.padded_shape(),
                    interm.dtype(),
                    interm.layout());
            }
            ttnn::experimental::ccl::validate_intermediate_tensor(
                input_tensor, interm, operation_attributes.optional_intermediate_mem_config);
        }
    }

    // Validate penult intermediate if provided. It only exists on the contiguous path, so it is invalid
    // both where that path does not apply and where a tiled intermediate opted out of it.
    if (tensor_args.optional_penult_intermediate_tensor.has_value()) {
        const auto& penult_interm = tensor_args.optional_penult_intermediate_tensor.value();
        TT_FATAL(
            use_contiguous,
            "A persistent penult intermediate was provided but this call does not use the contiguous "
            "reduce-scatter "
            "staging layout ({}); the penult intermediate is not applicable.",
            stage_spec.has_value() ? "the provided persistent intermediate selects the tiled layout"
                                   : "requires Ring topology and scatter dim != 0");
        auto penult_intermediate_spec = ttnn::experimental::ccl::reduce_scatter_ring_penult_intermediate_staging_spec(
            input_tensor,
            operation_attributes.topology,
            operation_attributes.dim,
            operation_attributes.ring_size,
            fp32_dest_acc_en);
        TT_FATAL(
            penult_intermediate_spec.has_value(),
            "penult intermediate staging spec must apply whenever the contiguous path is used");
        TT_FATAL(
            penult_interm.storage_type() == StorageType::DEVICE, "Persistent penult intermediate must be on device");
        TT_FATAL(
            penult_interm.device() == input_tensor.device(),
            "Persistent penult intermediate tensor must be allocated on the same mesh device as the input tensor");
        TT_FATAL(
            ttnn::experimental::ccl::reduce_scatter_tensor_matches_spec(penult_interm, *penult_intermediate_spec),
            "Persistent penult intermediate does not match the contiguous reduce-scatter penult intermediate "
            "staging layout "
            "(expected shape {}, UINT8 row-major interleaved DRAM). Allocate it with "
            "reduce_scatter_minimal_async_create_intermediate_buffer.",
            penult_intermediate_spec->logical_shape());
    }

    // Validate semaphore count
    constexpr auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Expected {} semaphores but got {}",
        num_expected_semaphores,
        operation_attributes.semaphore.size());
}

std::vector<tt::tt_metal::TensorSpec> ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    auto output_shape = input_tensor.logical_shape();
    output_shape[operation_attributes.dim] /= operation_attributes.ring_size;
    TensorSpec output_spec(
        output_shape,
        TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));

    // Contiguous ring fast path: the intermediate is a chunk-paged, row-major, interleaved-DRAM staging
    // buffer (page = one chunk = tile_granularity tiles). This lets the writer send whole chunks with a
    // single contiguous fused-unicast per packet instead of scatter-writes. The same spec is used to
    // allocate caller-provided persistent buffers (reduce_scatter_ring_interm_staging_spec). A
    // caller-provided input-shaped intermediate opts back into the tiled layout below.
    //
    // This path carries a third output: the smaller penult intermediate the 2nd-last ring
    // iteration writes into.
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    if (uses_contiguous_staging(operation_attributes, tensor_args)) {
        auto penult_intermediate_spec = ttnn::experimental::ccl::reduce_scatter_ring_penult_intermediate_staging_spec(
            input_tensor,
            operation_attributes.topology,
            operation_attributes.dim,
            operation_attributes.ring_size,
            fp32_dest_acc_en);
        TT_FATAL(
            penult_intermediate_spec.has_value(),
            "penult intermediate staging spec must apply whenever the contiguous path is used");
        return {
            *ttnn::experimental::ccl::reduce_scatter_ring_interm_staging_spec(
                input_tensor,
                operation_attributes.topology,
                operation_attributes.dim,
                operation_attributes.ring_size,
                fp32_dest_acc_en),
            output_spec,
            *penult_intermediate_spec};
    }

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
            adjusted_intermediate_mem_config = tt::tt_metal::MemoryConfig(
                intermediate_mem_config.memory_layout(),
                intermediate_mem_config.buffer_type(),
                intermediate_shard_spec);
        } else {
            adjusted_intermediate_mem_config = intermediate_mem_config;
        }
    } else {
        adjusted_intermediate_mem_config = intermediate_mem_config;
    }

    return {
        tt::tt_metal::TensorSpec(
            inter_shape,
            TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), adjusted_intermediate_mem_config)),
        output_spec,
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

    // Contiguous staging layout only (see compute_output_specs): the third spec is the penult
    // intermediate. Like the main intermediate, a caller-provided persistent buffer takes priority — it
    // was matched against this same spec in validate_on_program_cache_miss.
    if (tensor_specs.size() > 2) {
        ttnn::Tensor penult_intermediate_buffer = tensor_args.optional_penult_intermediate_tensor.has_value()
                                                      ? tensor_args.optional_penult_intermediate_tensor.value()
                                                      : create_device_tensor(tensor_specs[2], input_tensor.device());
        return {intermediate_buffer, output_buffer, penult_intermediate_buffer};
    }

    return {intermediate_buffer, output_buffer};
}

std::vector<tt::tt_metal::TensorTopology> ReduceScatterMinimalAsyncDeviceOperation::compute_output_topologies(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    // TODO(#48421): Enforce input invariants with TT_FATAL instead of sanitising malformed topologies.
    // Output is sharded along `dim` across `cluster_axis`; intermediate keeps the input topology.
    const auto& input_topology = tensor_args.input_tensor.tensor_topology();
    auto output_placements = input_topology.placements();

    // Normalize `dim` so Shard placements are stored consistently (avoids [Shard(-1), Shard(3)]).
    const int rank = static_cast<int>(tensor_args.input_tensor.logical_shape().rank());
    const int dim = ((static_cast<int>(operation_attributes.dim) % rank) + rank) % rank;
    const auto shard_placement = tt::tt_metal::distributed::MeshMapperConfig::Shard{dim};

    // Replicate any other mesh axis still sharding the same tensor dim — concat_ndim requires unique dims.
    auto clear_same_dim = [&](auto& placement) {
        if (auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&placement)) {
            if (((shard->dim % rank) + rank) % rank == dim) {
                placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
            }
        }
    };

    if (operation_attributes.cluster_axis.has_value()) {
        const auto axis = operation_attributes.cluster_axis.value();
        if (axis < output_placements.size()) {
            for (size_t i = 0; i < output_placements.size(); ++i) {
                if (i != axis) {
                    clear_same_dim(output_placements[i]);
                }
            }
            output_placements[axis] = shard_placement;
        }
    } else {
        // Whole-mesh reduce_scatter: shard only real (>1) axes on `dim`; replicate size-1 axes to keep dims unique.
        const auto& mesh_shape = input_topology.distribution_shape();
        for (size_t i = 0; i < output_placements.size(); ++i) {
            if (i < mesh_shape.dims() && mesh_shape[static_cast<int>(i)] > 1) {
                output_placements[i] = shard_placement;
            } else {
                output_placements[i] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
            }
        }
    }

    auto output_topology = tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(output_placements), input_topology.mesh_coords());

    // The count must match create_output_tensors exactly, so the contiguous staging path contributes a
    // third entry for the penult intermediate. It is device-local scratch with no distribution semantics of
    // its own, so it keeps the input topology — same as the intermediate at index 0.
    if (uses_contiguous_staging(operation_attributes, tensor_args)) {
        return {input_topology, std::move(output_topology), input_topology};
    }

    return {input_topology, std::move(output_topology)};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ReduceScatterMinimalAsyncDeviceOperation::tensor_return_value_t>
ReduceScatterMinimalAsyncDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    // =========================================================================
    // ReduceScatter Roofline Performance Model
    //
    // ReduceScatter: N devices, each has S bytes of input. Each device's
    // output is the element-wise reduction (sum) of its slice across all
    // devices: output size = S/N bytes per device.
    //
    // Unlike AllGather where data is forwarded unmodified (each link carries
    // full S-byte slices), ReduceScatter REDUCES data at each hop — each
    // link only carries S/N-byte slices. This makes ReduceScatter N× more
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
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    int clock_rate_mhz = 1000;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        arch = input_tensor.device()->arch();
        clock_rate_mhz = input_tensor.device()->get_clock_rate_mhz();
    }

    // --- Data sizes ---
    const uint64_t input_size_bytes =
        static_cast<uint64_t>(input_tensor.physical_volume()) * input_tensor.element_size();
    const uint32_t N = args.ring_size;
    const uint32_t num_links = args.num_links;
    const uint64_t slice_size = input_size_bytes / N;  // S/N bytes per device output

    // =========================================================================
    // 1. FABRIC TRANSFER (algorithm-agnostic lower bound)
    //
    // RING topology — bisection argument:
    //   Cut the ring into two halves of N/2 devices. Each device's output
    //   chunk needs contributions from devices on both sides of the cut.
    //   With optimal intermediate reduction, each direction carries
    //   (N-1)×S/(2N) bytes through the bottleneck link.
    //   Latency: N/2 hops (ring diameter — graph-theoretic minimum).
    //
    // LINE topology — endpoint bottleneck:
    //   The edge device (D0 or DN-1) has one link and must receive all
    //   (N-1) partial results through it: (N-1) × S/N bytes.
    //   Latency: N-1 hops (linear diameter).
    // =========================================================================
    uint64_t bottleneck_bytes = 0;  // bottleneck bytes through the most-loaded link
    uint32_t num_hops = 0;          // collective diameter (hops)
    if (N <= 1) {
        // Single device: no fabric communication
    } else if (args.topology == ttnn::ccl::Topology::Ring) {
        // Bisection lower bound: (N-1) * slice_size / 2 per direction
        bottleneck_bytes = tt::div_up((N - 1) * slice_size, 2);
        num_hops = N / 2;
    } else {
        // Line/Linear topology
        bottleneck_bytes = (N - 1) * slice_size;
        num_hops = N - 1;
    }
    // Fabric bandwidth and its pipeline-fill latency are two independent floors — both compete in the final max().
    const auto [fabric_bw_cycles, fabric_fill_cycles] = ttnn::ccl::estimate_fabric_transfer_cycles(
        arch, tt::tt_fabric::GetFabricConfig(), clock_rate_mhz, bottleneck_bytes, num_links, num_hops);

    // =========================================================================
    // 2. LOCAL DATA MOVEMENT — first-principles minimum (algorithm-agnostic)
    //
    // DRAM reads:  S bytes — each byte of the input tensor is read exactly
    //   once (to contribute to a reduction or to be sent to a neighbor).
    // DRAM writes: S/N bytes — only the final reduced output chunk.
    //   Intermediate partial results flow network → L1 → reduce → network
    //   without touching DRAM in the optimal case.
    //
    // Hardware ceiling: all device compute cores can drive DRAM concurrently.
    // DRAM read and write can overlap (different NOC channels), so we
    // take max(read_bw, write_bw) rather than summing them.
    // =========================================================================
    const auto& output_tensor = output_tensors[1];  // [0] is intermediate, [1] is output
    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const uint32_t read_page_size = input_tensor.buffer()->page_size();
    const uint32_t write_page_size = output_tensor.buffer()->page_size();
    const uint32_t device_rows = input_tensor.device()->compute_with_storage_grid_size().x;
    const uint32_t device_cols = input_tensor.device()->compute_with_storage_grid_size().y;
    const uint32_t num_cores = device_rows * device_cols;

    const uint64_t total_read_bytes = input_size_bytes;  // S
    const uint32_t read_pages_per_core = tt::div_up(tt::div_up(total_read_bytes, read_page_size), num_cores);

    const uint64_t total_write_bytes = slice_size;  // S/N
    const uint32_t write_pages_per_core = tt::div_up(tt::div_up(total_write_bytes, write_page_size), num_cores);

    auto [read_bw_cycles, read_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        read_page_size, input_is_dram, /*is_local=*/false, read_pages_per_core, arch, /*is_read=*/true);
    auto [write_bw_cycles, write_latency_cycles] = ttnn::operations::data_movement::get_cycles_for_transaction_size(
        write_page_size, output_is_dram, /*is_local=*/false, write_pages_per_core, arch, /*is_read=*/false);

    // =========================================================================
    // 3. COMPUTE (element-wise reduction) — algorithm-agnostic
    //
    // Each device performs (N-1) element-wise reductions (add_tiles),
    // regardless of reduction order (sequential, tree, etc.).
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
    //
    // Bottleneck is the unpacker (80 B/cycle joint SrcA+SrcB limit).
    // Distributed across num_cores (hardware ceiling).
    // =========================================================================
    constexpr uint32_t UNPACKER_BW_BYTES_PER_CYCLE = 80;
    const uint64_t total_unpack_bytes = 2ULL * (N - 1) * slice_size;
    const int compute_cycles =
        tt::div_up(total_unpack_bytes, static_cast<uint64_t>(num_cores) * UNPACKER_BW_BYTES_PER_CYCLE);

    // =========================================================================
    // 4. PIPELINED MODEL
    //
    // All resources operate in parallel, and a pipelined collective overlaps
    // fill/drain latency with steady-state streaming. So every term — bandwidth
    // AND fill/drain latency — compete; none stack on top:
    //   max( max(fabric_bw, dram_bw, compute), max(pipeline_latency, fabric_fill) )
    // pipeline_latency = read_latency (first DRAM read) + compute_latency
    //   (last chunk's reduction) + write_latency (last DRAM write).
    // =========================================================================
    const int local_bw_cycles = std::max(read_bw_cycles, write_bw_cycles);
    const int compute_latency_cycles =
        static_cast<int>(2ULL * slice_size / (static_cast<uint64_t>(num_cores) * UNPACKER_BW_BYTES_PER_CYCLE));
    const int pipeline_latency_cycles = read_latency_cycles + compute_latency_cycles + write_latency_cycles;
    const int throughput_cycles = std::max({local_bw_cycles, fabric_bw_cycles, compute_cycles});
    const int fill_cycles = std::max(pipeline_latency_cycles, fabric_fill_cycles);
    const int ideal_dev_clock_cycles = std::max(throughput_cycles, fill_cycles);

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
    const std::optional<ttnn::Tensor>& optional_penult_intermediate_tensor,
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
    auto tensor_args = OperationType::tensor_args_t{
        input_tensor, optional_intermediate_tensor, optional_output_tensor, optional_penult_intermediate_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "all_gather_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

void AllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto input_tensor = tensor_args.input_tensor;

    // Basic validations
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device!");
    TT_FATAL(input_tensor.logical_shape().rank() >= 2, "AllGather requires tensor of rank 2 or greater");

    uint32_t target_ring_size = ::ttnn::ccl::get_topological_dimension(input_tensor, operation_attributes.cluster_axis);
    TT_FATAL(target_ring_size > 1, "all_gather op will only work for num_devices > 1, but has {}", target_ring_size);

    TT_FATAL(
        operation_attributes.num_links > 0,
        "num_links should be more than 0 but has {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows");

    // Page alignment check
    auto page_size = input_tensor.buffer()->page_size();
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    // Memory layout validations
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    // Don't support input DRAM block sharding
    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().buffer_type() == BufferType::L1, "We don't support input DRAM block sharding");
    }

    validate_on_program_cache_hit(operation_attributes, tensor_args);
}

void AllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        auto output_tensor = tensor_args.optional_output_tensor.value();
        auto output_spec = compute_output_specs(operation_attributes, tensor_args);
        TT_FATAL(
            output_tensor.logical_shape() == output_spec.logical_shape(),
            "Output tensor shape {} does not match computed output spec shape {}",
            output_tensor.logical_shape(),
            output_spec.logical_shape());
        const auto& optional_output_tensor_spec = output_tensor.tensor_spec();
        // everything but memory config must match
        TT_FATAL(
            optional_output_tensor_spec.page_config() == output_spec.page_config(),
            "Output tensor page config {} does not match computed output spec page config {}",
            optional_output_tensor_spec.page_config(),
            output_spec.page_config());
        TT_FATAL(
            optional_output_tensor_spec.layout() == output_spec.layout(),
            "Output tensor layout {} does not match computed output spec layout {}",
            optional_output_tensor_spec.layout(),
            output_spec.layout());
        TT_FATAL(
            optional_output_tensor_spec.data_type() == output_spec.data_type(),
            "Output tensor data_type {} does not match computed output spec data_type {}",
            optional_output_tensor_spec.data_type(),
            output_spec.data_type());
        TT_FATAL(
            optional_output_tensor_spec.physical_shape() == output_spec.physical_shape(),
            "Output tensor physical shape {} does not match computed output spec physical shape {}",
            optional_output_tensor_spec.physical_shape(),
            output_spec.physical_shape());
        TT_FATAL(
            optional_output_tensor_spec.tile() == output_spec.tile(),
            "Output tensor tile {} does not match computed output spec tile {}",
            optional_output_tensor_spec.tile(),
            output_spec.tile());
    }
}

AllGatherDeviceOperation::spec_return_value_t AllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();
    uint32_t target_ring_size = ::ttnn::ccl::get_topological_dimension(input_tensor, operation_attributes.cluster_axis);
    output_shape[operation_attributes.dim] *= target_ring_size;

    auto mem_config = operation_attributes.memory_config;
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), mem_config));
}

AllGatherDeviceOperation::tensor_return_value_t AllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

AllGatherDeviceOperation::topology_return_value_t AllGatherDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();
    auto output_placements = input_topology.placements();

    // For each distribution dimension, if sharded on the gather dim, make it replicated
    for (auto& output_placement : output_placements) {
        if (auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&output_placement)) {
            if (shard->dim == static_cast<int>(operation_attributes.dim)) {
                output_placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
            }
        }
    }

    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), output_placements, input_topology.mesh_coords())};
}

ttsl::hash::hash_t AllGatherDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.subdevice_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        subdevice_core_range_set = subdevice_core_range_set.intersection(operation_attributes.sub_core_grid.value());
    }
    return tt::tt_metal::operation::hash_operation<AllGatherDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.cluster_axis,
        operation_attributes.memory_config,
        operation_attributes.topology,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        operation_attributes.use_l1_small_for_semaphores,
        subdevice_core_range_set,
        tensor_args);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<AllGatherDeviceOperation::tensor_return_value_t>
AllGatherDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    // =========================================================================
    // AllGather Roofline Performance Model (First-Principles)
    //
    // AllGather is pure data movement — no compute. The roofline is the
    // hardware ceiling: the minimum time given the physical topology and
    // memory bandwidth, independent of any particular algorithm.
    //
    // Performance is bounded by:
    //   ideal_cycles = max(DRAM_bw_cycles, fabric_bw_cycles) + pipeline_latency
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
    float clock_rate_ghz = 1.0f;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        arch = input_tensor.device()->arch();
        clock_rate_ghz = input_tensor.device()->get_clock_rate_mhz() / 1000.0f;
    }

    // Data size: bytes each device contributes
    const uint64_t input_size_bytes = input_tensor.physical_volume() * input_tensor.element_size();

    const uint32_t N = ::ttnn::ccl::get_topological_dimension(input_tensor, args.cluster_axis);
    const uint32_t num_links = args.num_links;
    double fabric_time_ns = 0.0f;
    if (N <= 1) {
        // Single device: no fabric communication
        fabric_time_ns = 0.0f;
    } else if (tt::tt_fabric::is_ring_or_torus(args.topology)) {
        // Ring topology: bisection cuts 2 links, so each direction carries
        // at most half the total data. Bottleneck per direction = ceil((N-1)*S/2).
        const uint64_t bottleneck_bytes = tt::div_up((N - 1) * input_size_bytes, 2);
        fabric_time_ns =
            ttnn::ccl::estimate_fabric_transfer_ns(arch, bottleneck_bytes, num_links, tt::div_up(N - 1, 2u));
    } else {
        // Line/Linear/Mesh topology: edge device has one link and must
        // receive all (N-1) slices through it.
        const uint64_t bottleneck_bytes = (N - 1) * input_size_bytes;
        fabric_time_ns = ttnn::ccl::estimate_fabric_transfer_ns(arch, bottleneck_bytes, num_links, N - 1);
    }

    // Convert fabric time (ns) to device clock cycles.
    // clock_rate_ghz cycles/ns * ns = cycles
    const int fabric_cycles = static_cast<int>(std::ceil(fabric_time_ns * clock_rate_ghz));

    // --- Local DRAM bandwidth ceiling (first-principles) ---
    // Read: device reads S bytes from DRAM.  Write: device writes N*S bytes.
    // Roofline assumes all device compute cores drive DRAM concurrently
    // (hardware max parallelism). BW competes with fabric (max); latencies
    // are additive (pipeline fill/drain).
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
    const int ideal_dev_clock_cycles = std::max(local_bw_cycles, fabric_cycles) + pipeline_latency_cycles;

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const std::optional<CoreRangeSet>& sub_core_grid,
    bool use_l1_small_for_semaphores) {
    using OperationType = ttnn::operations::ccl::AllGatherDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .memory_config = memory_config,
            .dim = dim,
            .cluster_axis = cluster_axis,
            .subdevice_id = subdevice_id,
            .topology = topology,
            .num_links = num_links,
            .chunks_per_sync = chunks_per_sync,
            .num_workers_per_link = num_workers_per_link,
            .num_buffers_per_channel = num_buffers_per_channel,
            .sub_core_grid = sub_core_grid,
            .use_l1_small_for_semaphores = use_l1_small_for_semaphores},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim

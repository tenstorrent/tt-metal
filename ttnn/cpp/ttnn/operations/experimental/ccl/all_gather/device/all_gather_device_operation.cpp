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

namespace ttnn::experimental::prim {

void AllGatherDeviceOperation::validate_on_program_cache_miss(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Constraints on input tensor
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffers on device!");

    // Constraints on other inputs
    int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    TT_FATAL(args.dim >= -rank && args.dim < rank, "Invalid gather dim {} for {}D input tensor", args.dim, rank);
    TT_FATAL(args.ring_size > 1, "all_gather collective will only work for num_devices > 1, got {}", args.ring_size);

    // If mesh_device shape is 2D but !FABRIC_2D, then must specify cluster_axis
    const auto mesh_shape = input_tensor.device()->shape();
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(::tt::tt_fabric::GetFabricConfig());
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
        TT_FATAL(
            output_tensor.memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
            "Output tensor memory layout {} should be same as input tensor memory layout {}",
            output_tensor.memory_config().memory_layout(),
            input_tensor.memory_config().memory_layout());

        // Check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        auto expected_output_shape = input_shape;
        expected_output_shape[args.dim] *= args.ring_size;
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

    // Implementation-specific constraints
    if (input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT && args.dim == rank - 1) {
        TT_FATAL(
            input_tensor.buffer()->page_size() == input_tensor.buffer()->aligned_page_size(),
            "all-gather requires aligned row-major pages for innermost gather dim");
    }
    if (input_tensor.layout() == ttnn::TILE_LAYOUT && args.dim == input_tensor.logical_shape().rank() - 1) {
        const auto& input_logical_shape = input_tensor.logical_shape();
        const auto& input_padded_shape = input_tensor.padded_shape();
        TT_FATAL(
            input_logical_shape[args.dim] == input_padded_shape[args.dim],
            "all-gather requires unpadded tiles along gather dim");
    }

    // Page-size relationship between input and output. Three copy modes (see the
    // "Page indexing" glossary in all_gather_factory.cpp for full definitions):
    //   matched (in == out): 1 write per input page, byte offset = 0.
    //   concat  (out > in) : 1 write per input page, byte offset = (d % concat_factor) * in.
    //   split   (in > out) : split_factor writes per input page, byte offset = 0.
    const auto output_spec = compute_output_specs(args, tensor_args);
    const uint32_t output_page_size = static_cast<uint32_t>(output_spec.compute_page_size_bytes());
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t num_devices = args.ring_size;

    TT_FATAL(
        std::max(output_page_size, input_page_size) % std::min(output_page_size, input_page_size) == 0,
        "all-gather: input/output page sizes ({}, {}) must have an integer ratio.",
        input_page_size,
        output_page_size);

    const uint32_t concat_factor = std::max(1u, output_page_size / input_page_size);
    const uint32_t split_factor = std::max(1u, input_page_size / output_page_size);

    // input_pages_per_row: # input pages along the page-defining dim per device.
    // >1 only for RM with multiple shards per device.
    uint32_t input_pages_per_row = 1;
    if (input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        const auto& shape = input_tensor.padded_shape();
        input_pages_per_row = (shape[shape.rank() - 1] * input_tensor.element_size()) / input_page_size;
    }

    // Concat mode needs concat_factor to divide num_devices (so N/concat_factor
    // output positions per row is integer).
    TT_FATAL(
        concat_factor == 1 || num_devices % concat_factor == 0,
        "all-gather: concat_factor={} must divide num_devices={}.",
        concat_factor,
        num_devices);

    // Concat mode with multi-shard input would need a different byte offset per
    // input page. The kernel currently takes a single byte_offset value, so reject.
    TT_FATAL(
        !(concat_factor > 1 && input_pages_per_row > 1),
        "all-gather: concat (concat_factor>1) with multi-shard input (input_pages_per_row>1) is not "
        "supported. concat_factor={}, input_pages_per_row={}.",
        concat_factor,
        input_pages_per_row);

    // Page sizes can only differ for RM last-dim gather (TILE pages are tile-sized;
    // non-last-dim gather doesn't change the page-defining dim).
    TT_FATAL(
        (concat_factor == 1 && split_factor == 1) ||
            (input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT && args.dim == rank - 1),
        "all-gather: differing page sizes (concat_factor={}, split_factor={}) only valid for RM "
        "last-dim gather.",
        concat_factor,
        split_factor);
}

AllGatherDeviceOperation::spec_return_value_t AllGatherDeviceOperation::compute_output_specs(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] *= args.ring_size;
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

AllGatherDeviceOperation::tensor_return_value_t AllGatherDeviceOperation::create_output_tensors(
    const AllGatherParams& args, const AllGatherInputs& tensor_args) {
    if (tensor_args.persistent_output_tensor.has_value()) {
        return tensor_args.persistent_output_tensor.value();
    }
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

ttsl::hash::hash_t AllGatherDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherDeviceOperation::compute_program_hash is called");

    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    return tt::tt_metal::operation::hash_operation<AllGatherDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_mem_config,
        args.topology,
        args.cluster_axis,
        tensor_args.persistent_output_tensor.has_value(),
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

    const uint32_t N = args.ring_size;
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

std::tuple<AllGatherParams, AllGatherInputs> all_gather_build_operation_args(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis) {
    // Query the machine and Fabric setup
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Input tensor should be on device for all_gather operation");
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis);
    auto topology = ::ttnn::ccl::get_usable_topology(input_tensor, std::nullopt, cluster_axis);
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    // Resolve negative gather dim
    uint32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    return {
        AllGatherParams(
            gather_dim,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            topology,
            cluster_axis),
        AllGatherInputs{.input_tensor = input_tensor, .persistent_output_tensor = persistent_output_tensor}};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor all_gather_experimental(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis) {
    auto [params, inputs] = experimental::prim::all_gather_build_operation_args(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis);
    return ttnn::device_operation::launch<experimental::prim::AllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn::prim

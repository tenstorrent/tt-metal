// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_device_operation.hpp"
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <ttnn/global_semaphore.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

void create_tensor_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_range_set,
    const ttnn::Tensor& tensor,
    uint32_t buffering_factor,
    tt::CBIndex cb_id,
    const std::string& tensor_name = "tensor") {
    auto page_size = get_page_size(tensor);
    auto num_pages = get_num_pages(tensor);
    auto aligned_page_size = get_aligned_page_size(tensor);
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    uint32_t cb_size = buffering_factor * aligned_page_size;

    log_debug(
        tt::LogOp,
        "{} shape: {}, pages: {}, page_size: {}, aligned_page_size: {} buffering_factor: {} cb_id: {} cb_size: {} "
        "cb_dtype: {}",
        tensor_name,
        tensor.logical_shape(),
        num_pages,
        page_size,
        aligned_page_size,
        buffering_factor,
        cb_id,
        cb_size,
        data_format);

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
}

}  // namespace detail

CombineProgramFactory::cached_mesh_workload_t CombineProgramFactory::create_mesh_workload(
    const CombineParams& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, CombineSharedVariables> shared_variables;

    auto* mesh_device = tensor_args.dispatched_buffer.device();

    // Create global semaphore for cross-device synchronization (fabric init barrier)
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    // Create a program for each device in the mesh
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes, coord, tensor_args, tensor_return_value, tensor_coords, init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<CombineSharedVariables> CombineProgramFactory::create_at(
    const CombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore) {
    tt::tt_metal::Program program{};

    // Extract input tensors
    const auto& dispatched_buffer = tensor_args.dispatched_buffer;
    const auto& dispatched_metadata = tensor_args.dispatched_metadata;
    const auto& expert_token_counts = tensor_args.expert_token_counts;
    const auto& output_tensor = tensor_return_value;

    auto* mesh_device = dispatched_buffer.device();
    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    // Extract mesh information for this specific device
    const auto& mesh_view = mesh_device->get_view();
    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
    uint32_t mesh_rows = mesh_view.num_rows();
    uint32_t mesh_cols = mesh_view.num_cols();

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    log_debug(
        tt::LogOp,
        "Creating prefill combine program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized: {} mesh shape: ({}, {}) topology: {} num_links: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord,
        mesh_rows,
        mesh_cols,
        topology,
        num_links);

    // Get fabric configuration and connection information
    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // Get neighbor connections for fabric
    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // Figure out tensor dimensions
    auto dispatched_shape = dispatched_buffer.logical_shape();
    auto hidden_size = dispatched_shape[-1];
    auto max_dispatched_tokens_per_expert = dispatched_shape[-2];

    // Calculate work distribution
    // For combine, we iterate over (chips, experts, tokens_per_expert)
    // Start with single-core implementation, can parallelize later
    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    TT_FATAL(!subdevice_cores.empty(), "Need at least one core for combine operation");

    // Use first available core for now (single-core implementation)
    CoreCoord worker_core = subdevice_cores.at(0);
    CoreRangeSet worker_core_grid = CoreRangeSet({CoreRange(worker_core, worker_core)});

    // Create local semaphore for reader→writer zero-init synchronization (same core, no cross-device)
    auto zero_init_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_grid, 0);

    log_debug(tt::LogOp, "Creating prefill combine program with hidden_size: {} on core: {}", hidden_size, worker_core);

    // Create input CBs (readers)
    detail::create_tensor_cb(
        program,
        worker_core_grid,
        dispatched_buffer,
        /*buffering_factor=*/2,
        /*cb_id=*/tt::CBIndex::c_0,
        "dispatched_buffer");

    detail::create_tensor_cb(
        program,
        worker_core_grid,
        dispatched_metadata,
        /*buffering_factor=*/2,
        /*cb_id=*/tt::CBIndex::c_1,
        "dispatched_metadata");

    detail::create_tensor_cb(
        program,
        worker_core_grid,
        expert_token_counts,
        /*buffering_factor=*/detail::get_num_pages(expert_token_counts),  // Read entire counter
        /*cb_id=*/tt::CBIndex::c_2,
        "expert_token_counts");

    // Create output CB (writer)
    detail::create_tensor_cb(
        program,
        worker_core_grid,
        output_tensor,
        /*buffering_factor=*/2,
        /*cb_id=*/tt::CBIndex::c_3,
        "output_tensor");

    // Create packet header CB for fabric communication (if fabric is enabled)
    if (num_links > 0) {
        constexpr uint32_t num_packet_headers = 1;  // Only need unicast header for combine
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_4, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_4, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, worker_core_grid, packet_header_cb_config);

        log_debug(
            tt::LogOp,
            "Created packet header CB with size {} bytes for {} headers",
            packet_header_cb_size,
            num_packet_headers);
    }

    // Generate destination chip and mesh IDs for fabric connections
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", ccl::common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", ccl::common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", ccl::common::stringify(directions));

    // Create reader kernel compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        // CB IDs (4)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_dispatched_buffer_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_dispatched_metadata_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_expert_token_counts_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_output_id

        // Page counts (4)
        detail::get_num_pages(dispatched_buffer),
        detail::get_num_pages(dispatched_metadata),
        detail::get_num_pages(expert_token_counts),
        detail::get_num_pages(output_tensor),

        // Page sizes (4)
        detail::get_page_size(dispatched_buffer),
        detail::get_page_size(dispatched_metadata),
        detail::get_page_size(expert_token_counts),
        detail::get_page_size(output_tensor),

        // Operation parameters (5)
        operation_attributes.dispatch_group_size,
        operation_attributes.experts_per_chip,
        operation_attributes.num_experts_per_tok,
        operation_attributes.seq_len_per_chip,
        (uint32_t)max_dispatched_tokens_per_expert,

        // Hidden dimension
        (uint32_t)hidden_size,

        // Aligned page sizes (4)
        detail::get_aligned_page_size(dispatched_buffer),
        detail::get_aligned_page_size(dispatched_metadata),
        detail::get_aligned_page_size(expert_token_counts),
        detail::get_aligned_page_size(output_tensor),

        // Mesh information (5)
        src_mesh_id,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        linearized_mesh_coord,

        // Fabric configuration (4)
        (uint32_t)fabric_max_packet_size,
        l1_alignment,
        num_links,
        static_cast<uint32_t>(topology),
    };

    // Append TensorAccessorArgs for all tensors
    tt::tt_metal::TensorAccessorArgs(dispatched_buffer.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatched_metadata.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(expert_token_counts.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_time_args);

    // Generate defines for reader kernel
    std::map<std::string, std::string> reader_defines;
    reader_defines["INIT_ZEROS"] = operation_attributes.init_zeros ? "1" : "0";

    // Check if output is L1 interleaved and add multicast defines
    bool is_l1_output = output_tensor.buffer()->buffer_type() == BufferType::L1;
    reader_defines["IS_L1_OUTPUT"] = is_l1_output ? "1" : "0";

    if (is_l1_output && operation_attributes.init_zeros) {
        // Get compute_with_storage_grid for L1 bank cores
        auto compute_grid = mesh_device->compute_with_storage_grid_size();
        uint32_t num_l1_banks = compute_grid.x * compute_grid.y;

        // Get NOC coordinates for the bank core grid
        // Start is (0,0), end is (grid_x-1, grid_y-1) in logical coords
        // Convert to virtual/NOC coords
        CoreCoord logical_start(0, 0);
        CoreCoord logical_end(compute_grid.x - 1, compute_grid.y - 1);
        CoreCoord noc_start = mesh_device->virtual_core_from_logical_core(logical_start, tt::CoreType::WORKER);
        CoreCoord noc_end = mesh_device->virtual_core_from_logical_core(logical_end, tt::CoreType::WORKER);

        // Calculate bytes per bank based on lock-step allocator behavior:
        // Each bank reserves space for ceil(num_pages / num_banks) pages
        uint32_t num_pages = detail::get_num_pages(output_tensor);
        uint32_t aligned_page_size = detail::get_aligned_page_size(output_tensor);
        uint32_t pages_per_bank = (num_pages + num_l1_banks - 1) / num_l1_banks;
        uint32_t bytes_per_bank = pages_per_bank * aligned_page_size;

        reader_defines["NUM_L1_BANKS"] = std::to_string(num_l1_banks);
        reader_defines["OUTPUT_BYTES_PER_BANK"] = std::to_string(bytes_per_bank);
        reader_defines["L1_BANK_NOC_X_START"] = std::to_string(noc_start.x);
        reader_defines["L1_BANK_NOC_Y_START"] = std::to_string(noc_start.y);
        reader_defines["L1_BANK_NOC_X_END"] = std::to_string(noc_end.x);
        reader_defines["L1_BANK_NOC_Y_END"] = std::to_string(noc_end.y);

        log_debug(
            tt::LogOp,
            "L1 multicast zero-init: num_banks={} bytes_per_bank={} grid=({},{}) to ({},{})",
            num_l1_banks,
            bytes_per_bank,
            noc_start.x,
            noc_start.y,
            noc_end.x,
            noc_end.y);
    }

    // Create reader kernel
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp",
        worker_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    // Create writer kernel - shares same compile time args
    const auto& writer_compile_time_args = reader_compile_time_args;

    // Generate fabric defines for writer kernel
    std::map<std::string, std::string> writer_defines;
    if (num_links > 0) {
        writer_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        writer_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        writer_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp",
        worker_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    // Set runtime args
    // Base tensor addresses (shared between reader and writer)
    std::vector<uint32_t> base_runtime_args = {
        dispatched_buffer.buffer()->address(),
        dispatched_metadata.buffer()->address(),
        expert_token_counts.buffer()->address(),
        output_tensor.buffer()->address(),
    };

    // Reader needs: base + zero_init semaphore ID (kernel uses get_semaphore() to convert to address)
    std::vector<uint32_t> reader_runtime_args = base_runtime_args;
    reader_runtime_args.push_back(zero_init_semaphore_id);

    // Writer needs: base + zero_init semaphore ID + init_semaphore address (global)
    std::vector<uint32_t> writer_runtime_args = base_runtime_args;
    writer_runtime_args.push_back(zero_init_semaphore_id);
    writer_runtime_args.push_back((uint32_t)init_semaphore.address());

    // Append fabric connection args (only if fabric is enabled)
    if (num_links > 0) {
        for (const auto& neighbor_coordinate : neighbors) {
            // Skip self-connections
            if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                log_debug(
                    tt::LogOp,
                    "Skipping self-connection for mesh coord ({}, {}) at core {}",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    worker_core);
                continue;
            }

            log_debug(
                tt::LogOp,
                "Connection between mesh coord ({}, {}) and ({}, {}) at core {}",
                mesh_coordinate[0],
                mesh_coordinate[1],
                neighbor_coordinate[0],
                neighbor_coordinate[1],
                worker_core);

            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id,
                mesh_device->get_fabric_node_id(neighbor_coordinate),
                0,  // link_id - use 0 for single link
                program,
                worker_core,
                writer_runtime_args);
        }
    }

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, worker_core, reader_runtime_args);
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, worker_core, writer_runtime_args);

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .worker_core = worker_core,
         .init_semaphore = init_semaphore,
         .zero_init_semaphore_id = zero_init_semaphore_id}};
}

void CombineProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const CombineParams& /*operation_attributes*/,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    // Update buffer addresses in runtime args when tensors are reallocated
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        std::vector<uint32_t> reader_runtime_args = {
            tensor_args.dispatched_buffer.buffer()->address(),
            tensor_args.dispatched_metadata.buffer()->address(),
            tensor_args.expert_token_counts.buffer()->address(),
            tensor_return_value.buffer()->address(),
            shared_variables.zero_init_semaphore_id,
        };

        std::vector<uint32_t> writer_runtime_args = {
            tensor_args.dispatched_buffer.buffer()->address(),
            tensor_args.dispatched_metadata.buffer()->address(),
            tensor_args.expert_token_counts.buffer()->address(),
            tensor_return_value.buffer()->address(),
        };
        writer_runtime_args.push_back(shared_variables.zero_init_semaphore_id);
        writer_runtime_args.push_back((uint32_t)shared_variables.init_semaphore.address());
        // Note: Fabric connection args are not updated here as they don't change

        tt::tt_metal::SetRuntimeArgs(
            program, shared_variables.reader_kernel_id, shared_variables.worker_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(
            program, shared_variables.writer_kernel_id, shared_variables.worker_core, writer_runtime_args);
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine

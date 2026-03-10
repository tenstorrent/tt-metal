// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_combine_device_operation.hpp"
#include "prefill_combine_program_factory.hpp"
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

namespace ttnn::operations::experimental::deepseek::prefill_combine {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) {
    return (uint32_t)tensor.buffer()->num_pages();
}

uint32_t get_page_size(const ttnn::Tensor& tensor) {
    return (uint32_t)tensor.buffer()->page_size();
}

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) {
    return (uint32_t)tensor.buffer()->aligned_page_size();
}

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
        "{} shape: {}, pages: {}, page_size: {}, aligned_page_size: {} buffering_factor: {} cb_id: {} cb_size: {} cb_dtype: {}",
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
        tt::tt_metal::CircularBufferConfig(cb_size, {{cb_id, data_format}})
            .set_page_size(cb_id, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
}

}  // namespace detail

PrefillCombineDeviceOperation::PrefillCombineProgramFactory::cached_mesh_workload_t
PrefillCombineDeviceOperation::PrefillCombineProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.dispatched_buffer.device();

    // Create global semaphores for cross-device synchronization
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    // Create a program for each device in the mesh
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            tensor_coords,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<PrefillCombineDeviceOperation::PrefillCombineProgramFactory::shared_variables_t>
PrefillCombineDeviceOperation::PrefillCombineProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {

    tt::tt_metal::Program program{};

    // Extract input tensors
    const auto& dispatched_buffer = tensor_args.dispatched_buffer;
    const auto& dispatched_metadata = tensor_args.dispatched_metadata;
    const auto& experts_tok_counter = tensor_args.experts_tok_counter;
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

    // Calculate work distribution: split local experts across cores
    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    uint32_t effective_num_links = num_links >= 2 ? 2u : 1u;
    TT_FATAL(
        subdevice_cores.size() >= effective_num_links,
        "Not enough cores {} for {} links",
        subdevice_cores.size(),
        effective_num_links);

    uint32_t num_cores = effective_num_links;
    uint32_t experts_per_core_range = tt::div_up(operation_attributes.experts_per_chip, effective_num_links);

    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    log_debug(
        tt::LogOp,
        "Creating prefill combine program with hidden_size: {} num_cores: {} experts_per_core_range: {} cores: {}",
        hidden_size,
        num_cores,
        experts_per_core_range,
        sender_cores);

    // Create input CBs (readers)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatched_buffer,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_0,
        "dispatched_buffer");

    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatched_metadata,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_1,
        "dispatched_metadata");

    detail::create_tensor_cb(
        program,
        sender_core_grid,
        experts_tok_counter,
        /*buffering_factor=*/detail::get_num_pages(experts_tok_counter),  // Read entire counter
        /*cb_id=*/tt::CBIndex::c_2,
        "experts_tok_counter");

    // Create output CB (writer)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
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
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);

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
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_experts_tok_counter_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_output_id

        // Page counts (4)
        detail::get_num_pages(dispatched_buffer),
        detail::get_num_pages(dispatched_metadata),
        detail::get_num_pages(experts_tok_counter),
        detail::get_num_pages(output_tensor),

        // Page sizes (4)
        detail::get_page_size(dispatched_buffer),
        detail::get_page_size(dispatched_metadata),
        detail::get_page_size(experts_tok_counter),
        detail::get_page_size(output_tensor),

        // Operation parameters (5)
        operation_attributes.num_chips,
        operation_attributes.experts_per_chip,
        operation_attributes.num_experts_per_tok,
        operation_attributes.seq_len_per_chip,
        (uint32_t)max_dispatched_tokens_per_expert,

        // Hidden dimension
        (uint32_t)hidden_size,

        // Aligned page sizes (4)
        detail::get_aligned_page_size(dispatched_buffer),
        detail::get_aligned_page_size(dispatched_metadata),
        detail::get_aligned_page_size(experts_tok_counter),
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
        std::min(num_links, 1u),
        static_cast<uint32_t>(topology),
    };

    // Append TensorAccessorArgs for all tensors
    tt::tt_metal::TensorAccessorArgs(dispatched_buffer.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatched_metadata.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(experts_tok_counter.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_time_args);

    // Create reader kernel
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/prefill_combine/device/kernels/dataflow/"
        "reader_prefill_combine.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = reader_compile_time_args});

    // Create writer kernel - shares same compile time args
    const auto& writer_compile_time_args = reader_compile_time_args;

    // Generate fabric defines for writer kernel
    std::map<std::string, std::string> writer_defines;
    if (num_links > 0) {
        writer_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        writer_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        writer_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/prefill_combine/device/kernels/dataflow/"
        "writer_prefill_combine.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    // Set runtime args — each core handles a disjoint local expert range and its own fabric link
    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        // Expert range for this core (local expert indices)
        uint32_t expert_start = core_idx * experts_per_core_range;
        uint32_t expert_end = std::min((core_idx + 1) * experts_per_core_range, operation_attributes.experts_per_chip);

        std::vector<uint32_t> reader_runtime_args = {
            dispatched_buffer.buffer()->address(),
            dispatched_metadata.buffer()->address(),
            experts_tok_counter.buffer()->address(),
            output_tensor.buffer()->address(),
            expert_start,
            expert_end,
        };

        std::vector<uint32_t> writer_runtime_args = {
            dispatched_buffer.buffer()->address(),
            dispatched_metadata.buffer()->address(),
            experts_tok_counter.buffer()->address(),
            output_tensor.buffer()->address(),
            (uint32_t)cross_device_semaphore.address(),
            (uint32_t)init_semaphore.address(),
            expert_start,
            expert_end,
        };

        // Append fabric connection args (only if fabric is enabled)
        if (num_links > 0) {
            uint32_t core_link = core_idx % num_links;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    log_debug(
                        tt::LogOp,
                        "Skipping self-connection for mesh coord ({}, {}) at core {}",
                        mesh_coordinate[0],
                        mesh_coordinate[1],
                        sender_core);
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Connection between mesh coord ({}, {}) and ({}, {}) at core {} link {} "
                    "experts [{}, {})",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    core_link,
                    expert_start,
                    expert_end);

                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core, writer_runtime_args);
        core_idx++;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void PrefillCombineDeviceOperation::PrefillCombineProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        for (const auto& core : shared_variables.cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);

            // Update buffer addresses (indices 0-3)
            reader_runtime_args.at(0) = tensor_args.dispatched_buffer.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.dispatched_metadata.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.experts_tok_counter.buffer()->address();
            reader_runtime_args.at(3) = tensor_return_value.buffer()->address();

            writer_runtime_args.at(0) = tensor_args.dispatched_buffer.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.dispatched_metadata.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.experts_tok_counter.buffer()->address();
            writer_runtime_args.at(3) = tensor_return_value.buffer()->address();
            writer_runtime_args.at(4) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(5) = (uint32_t)shared_variables.init_semaphore.address();
            // Note: expert ranges (indices 6-7) and fabric args remain unchanged
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine

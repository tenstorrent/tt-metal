// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch_combined_device_operation.hpp"
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/global_semaphore.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_dispatch_combined {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

void create_tensor_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_range_set,
    const ttnn::Tensor& tensor,
    uint32_t buffering_factor,
    tt::CBIndex cb_id,
    const std::string& tensor_name = "tensor") {
    auto page_size = get_page_size(tensor);
    auto aligned_page_size = get_aligned_page_size(tensor);
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    uint32_t cb_size = buffering_factor * aligned_page_size;

    log_debug(
        tt::LogOp,
        "{} shape: {}, page_size: {}, aligned_page_size: {} buffering_factor: {} cb_id: {} cb_size: {}",
        tensor_name,
        tensor.logical_shape(),
        page_size,
        aligned_page_size,
        buffering_factor,
        cb_id,
        cb_size);

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
}

}  // namespace detail

PrefillDispatchCombinedDeviceOperation::PrefillDispatchCombinedProgramFactory::cached_mesh_workload_t
PrefillDispatchCombinedDeviceOperation::PrefillDispatchCombinedProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

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

ttnn::device_operation::CachedProgram<
    PrefillDispatchCombinedDeviceOperation::PrefillDispatchCombinedProgramFactory::shared_variables_t>
PrefillDispatchCombinedDeviceOperation::PrefillDispatchCombinedProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.indices_tensor;
    auto weights_tensor = tensor_args.weights_tensor;
    auto offsets_tensor = tensor_args.chip_to_n_routed_expert_offset_tensor;

    const auto& combined_output_tensor = tensor_return_value.at(0);
    const auto& experts_counter_tensor = tensor_return_value.at(1);

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
    log_debug(
        tt::LogOp,
        "Creating combined dispatch program for mesh coordinate: ({}, {}) with topology: {} num_links: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        topology,
        num_links);

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;
    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    uint32_t effective_num_links = num_links >= 2 ? 2u : 1u;
    uint32_t num_cores = (num_links > 0) ? effective_num_links * 2 : 1;
    TT_FATAL(
        subdevice_cores.size() >= num_cores,
        "Not enough cores {} for {} cores ({} links)",
        subdevice_cores.size(),
        num_cores,
        effective_num_links);

    auto logical_volume = input_tensor.logical_shape().volume();
    auto hidden_size = input_tensor.logical_shape()[-1];
    auto tokens_per_device = logical_volume / hidden_size;

    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    std::vector<CoreCoord> sender_cores_physical;
    for (const auto& core : sender_cores) {
        sender_cores_physical.push_back(mesh_device->worker_core_from_logical_core(core));
    }
    uint32_t num_secondary_cores = (num_cores > 1) ? (num_cores - 1) : 0;

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    uint32_t metadata_bytes = operation_attributes.metadata_len * sizeof(int32_t);
    uint32_t padded_metadata_bytes = tt::round_up(metadata_bytes, l1_alignment);

    // Combined CB: metadata padding + input payload per page
    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    uint32_t combined_cb_page_size = padded_metadata_bytes + aligned_input_page_size;
    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    log_debug(
        tt::LogOp,
        "Combined CB: padded_metadata_bytes={} aligned_input_page_size={} combined_cb_page_size={}",
        padded_metadata_bytes,
        aligned_input_page_size,
        combined_cb_page_size);

    constexpr uint32_t combined_cb_buffering = 16;
    tt::tt_metal::CircularBufferConfig combined_cb_config =
        tt::tt_metal::CircularBufferConfig(
            combined_cb_buffering * combined_cb_page_size, {{tt::CBIndex::c_0, input_data_format}})
            .set_page_size(tt::CBIndex::c_0, combined_cb_page_size);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, combined_cb_config);

    // Indices and weights CBs (reader-only scratch, batch-buffered for grouped DRAM reads)
    constexpr uint32_t read_batch_size = 8;
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        tt::CBIndex::c_1,
        "indices_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        tt::CBIndex::c_2,
        "weights_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        offsets_tensor,
        /*buffering_factor=*/detail::get_num_pages(offsets_tensor),
        tt::CBIndex::c_3,
        "offsets_tensor");

    // Combined output CB (for infrastructure)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        combined_output_tensor,
        /*buffering_factor=*/16,
        tt::CBIndex::c_4,
        "combined_output");
    detail::create_tensor_cb(
        program, sender_core_grid, experts_counter_tensor, /*buffering_factor=*/16, tt::CBIndex::c_5, "counter");

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // Active direction indices in the order get_neighbors adds them
    // (matches the writer's dir=0..3 iteration order)
    std::vector<uint32_t> active_dir_indices;
    {
        uint32_t axis_val = operation_attributes.axis.value_or(0);
        if (axis_val == 1) {
            if (directions[0]) {
                active_dir_indices.push_back(0);  // East
            }
            if (directions[1]) {
                active_dir_indices.push_back(1);  // West
            }
        } else {
            if (directions[2]) {
                active_dir_indices.push_back(2);  // North
            }
            if (directions[3]) {
                active_dir_indices.push_back(3);  // South
            }
        }
    }

    constexpr uint32_t DIR_INVALID = 0xFFFFFFFF;

    auto get_core_direction = [&](uint32_t cidx) -> uint32_t {
        if (num_links == 0) {
            return DIR_INVALID;
        }
        uint32_t dir_slot = cidx % 2;
        return (dir_slot < active_dir_indices.size()) ? active_dir_indices[dir_slot] : DIR_INVALID;
    };
    auto get_core_link = [&](uint32_t cidx) -> uint32_t { return cidx / 2; };

    // Multicast bounding box for secondary cores (cores 1..N-1)
    uint32_t mcast_x_start = 0, mcast_y_start = 0, mcast_x_end = 0, mcast_y_end = 0;
    if (num_secondary_cores > 0) {
        mcast_x_start = sender_cores_physical[1].x;
        mcast_y_start = sender_cores_physical[1].y;
        mcast_x_end = sender_cores_physical[1].x;
        mcast_y_end = sender_cores_physical[1].y;
        for (uint32_t i = 2; i < num_cores; i++) {
            mcast_x_start = std::min(mcast_x_start, (uint32_t)sender_cores_physical[i].x);
            mcast_y_start = std::min(mcast_y_start, (uint32_t)sender_cores_physical[i].y);
            mcast_x_end = std::max(mcast_x_end, (uint32_t)sender_cores_physical[i].x);
            mcast_y_end = std::max(mcast_y_end, (uint32_t)sender_cores_physical[i].y);
        }
    }

    // Packet header CB for fabric sends
    if (operation_attributes.num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_6, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_6, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    }

    // Route info CB for reader -> writer communication (4 x uint32_t per entry)
    uint32_t route_info_page_size = l1_alignment;
    constexpr uint32_t route_info_buffering = 16;
    tt::tt_metal::CircularBufferConfig route_info_cb_config =
        tt::tt_metal::CircularBufferConfig(
            route_info_buffering * route_info_page_size, {{tt::CBIndex::c_7, tt::DataFormat::UInt8}})
            .set_page_size(tt::CBIndex::c_7, route_info_page_size);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, route_info_cb_config);

    // Scratch CB for reader-side combined page staging (batch-buffered for grouped DRAM reads)
    tt::tt_metal::CircularBufferConfig scratch_cb_config =
        tt::tt_metal::CircularBufferConfig(
            read_batch_size * combined_cb_page_size, {{tt::CBIndex::c_8, input_data_format}})
            .set_page_size(tt::CBIndex::c_8, combined_cb_page_size);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, scratch_cb_config);

    // Semaphores for L1 shared read synchronization between primary/secondary reader cores
    const uint32_t batch_ready_sem_id = tt::tt_metal::CreateSemaphore(program, sender_core_grid, 0);
    const uint32_t batch_consumed_sem_id = tt::tt_metal::CreateSemaphore(program, sender_core_grid, 0);

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    // Compile-time args shared between reader and writer
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (5)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_combined_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_indices_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_weights_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_offsets_id
        static_cast<uint32_t>(tt::CBIndex::c_6),  // cb_packet_header_id

        // Page counts (6)
        detail::get_num_pages(input_tensor),
        detail::get_num_pages(indices_tensor),
        detail::get_num_pages(weights_tensor),
        detail::get_num_pages(offsets_tensor),
        detail::get_num_pages(combined_output_tensor),
        detail::get_num_pages(experts_counter_tensor),

        // Page sizes (6)
        detail::get_page_size(input_tensor),
        detail::get_page_size(indices_tensor),
        detail::get_page_size(weights_tensor),
        detail::get_page_size(offsets_tensor),
        detail::get_page_size(combined_output_tensor),
        detail::get_page_size(experts_counter_tensor),

        // Operation parameters (9)
        mesh_view.num_devices(),
        (uint32_t)hidden_size,
        operation_attributes.experts_per_chip,
        operation_attributes.n_routed_experts,
        operation_attributes.num_experts_per_tok,
        operation_attributes.metadata_len,
        operation_attributes.max_dispatched_tokens_per_expert,
        (uint32_t)tokens_per_device,
        padded_metadata_bytes,

        // Mesh information (5)
        src_mesh_id,
        src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        linearized_mesh_coord,

        // Aligned page sizes (6)
        detail::get_aligned_page_size(input_tensor),
        detail::get_aligned_page_size(indices_tensor),
        detail::get_aligned_page_size(weights_tensor),
        detail::get_aligned_page_size(offsets_tensor),
        detail::get_aligned_page_size(combined_output_tensor),
        detail::get_aligned_page_size(experts_counter_tensor),

        // Fabric configuration (4)
        (uint32_t)fabric_max_packet_size,
        l1_alignment,
        (uint32_t)std::min(operation_attributes.num_links, 1u),
        static_cast<uint32_t>(topology),

        // Additional CB IDs (2) - indices 41-42
        static_cast<uint32_t>(tt::CBIndex::c_7),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_8),  // cb_scratch_id
    };

    // Append TensorAccessorArgs for all 6 tensors (starting at index 43)
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weights_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(combined_output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(experts_counter_tensor.buffer()).append_to(compile_time_args);

    std::map<std::string, std::string> reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/prefill_dispatch_combined/device/kernels/dataflow/"
        "reader_prefill_dispatch_combined.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = reader_defines});

    std::map<std::string, std::string> writer_defines;
    if (operation_attributes.num_links > 0) {
        writer_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        writer_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        writer_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }
    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/prefill_dispatch_combined/device/kernels/dataflow/"
        "writer_prefill_dispatch_combined.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = writer_defines});

    // Runtime args (shared layout between reader and writer)
    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),            // 0
        indices_tensor.buffer()->address(),          // 1
        weights_tensor.buffer()->address(),          // 2
        offsets_tensor.buffer()->address(),          // 3
        combined_output_tensor.buffer()->address(),  // 4
        experts_counter_tensor.buffer()->address(),  // 5
        (uint32_t)cross_device_semaphore.address(),  // 6
        (uint32_t)init_semaphore.address(),          // 7
        0,                                           // token_start_idx                     // 8
        0,                                           // token_end_idx                       // 9
        0,                                           // dispatch_core_idx                   // 10
        0,                                           // num_dispatch_cores                  // 11
        0,                                           // assigned_direction                  // 12
        0,                                           // assigned_link                       // 13
        batch_ready_sem_id,                          // 14
        batch_consumed_sem_id,                       // 15
        num_secondary_cores,                         // 16
        0,                                           // primary_noc_x                       // 17
        0,                                           // primary_noc_y                       // 18
        mcast_x_start,                               // 19
        mcast_y_start,                               // 20
        mcast_x_end,                                 // 21
        mcast_y_end,                                 // 22
    };

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> writer_runtime_args = reader_runtime_args;

        reader_runtime_args[8] = 0;
        reader_runtime_args[9] = (uint32_t)tokens_per_device;
        writer_runtime_args[8] = 0;
        writer_runtime_args[9] = (uint32_t)tokens_per_device;

        reader_runtime_args[10] = core_idx;
        reader_runtime_args[11] = num_cores;
        writer_runtime_args[10] = core_idx;
        writer_runtime_args[11] = num_cores;

        uint32_t core_direction = get_core_direction(core_idx);
        uint32_t core_link = get_core_link(core_idx);
        reader_runtime_args[12] = core_direction;
        reader_runtime_args[13] = core_link;
        writer_runtime_args[12] = core_direction;
        writer_runtime_args[13] = core_link;

        reader_runtime_args[17] = sender_cores_physical[0].x;
        reader_runtime_args[18] = sender_cores_physical[0].y;
        writer_runtime_args[17] = sender_cores_physical[0].x;
        writer_runtime_args[18] = sender_cores_physical[0].y;

        if (operation_attributes.num_links > 0) {
            if (core_idx == 0) {
                // Core 0: connections for ALL neighbors (used for init sync);
                // appended in direction order to match writer's build_from_args loop
                for (uint32_t dir = 0; dir < 4; dir++) {
                    if (!directions[dir]) {
                        continue;
                    }
                    size_t neighbor_idx = 0;
                    for (size_t j = 0; j < active_dir_indices.size(); j++) {
                        if (active_dir_indices[j] == dir) {
                            neighbor_idx = j;
                            break;
                        }
                    }
                    if (neighbor_idx < neighbors.size()) {
                        const auto& nc = neighbors[neighbor_idx];
                        if (!(nc[0] == mesh_coordinate[0] && nc[1] == mesh_coordinate[1])) {
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_fabric_node_id,
                                mesh_device->get_fabric_node_id(nc),
                                core_link,
                                program,
                                sender_core,
                                writer_runtime_args);
                        }
                    }
                }
            } else {
                // Cores 1+: connection for assigned direction's neighbor only
                if (core_direction != DIR_INVALID) {
                    size_t neighbor_idx = 0;
                    for (size_t j = 0; j < active_dir_indices.size(); j++) {
                        if (active_dir_indices[j] == core_direction) {
                            neighbor_idx = j;
                            break;
                        }
                    }
                    if (neighbor_idx < neighbors.size()) {
                        const auto& nc = neighbors[neighbor_idx];
                        if (!(nc[0] == mesh_coordinate[0] && nc[1] == mesh_coordinate[1])) {
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_fabric_node_id,
                                mesh_device->get_fabric_node_id(nc),
                                core_link,
                                program,
                                sender_core,
                                writer_runtime_args);
                        }
                    }
                }
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

void PrefillDispatchCombinedDeviceOperation::PrefillDispatchCombinedProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& reader_kernel_id = shared_variables.reader_kernel_id;
        const auto& writer_kernel_id = shared_variables.writer_kernel_id;
        const auto& cores = shared_variables.cores;

        const auto& combined_output_tensor = tensor_return_value.at(0);
        const auto& experts_counter_tensor = tensor_return_value.at(1);

        for (const auto& core : cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);

            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            reader_runtime_args.at(3) = tensor_args.chip_to_n_routed_expert_offset_tensor.buffer()->address();
            reader_runtime_args.at(4) = combined_output_tensor.buffer()->address();
            reader_runtime_args.at(5) = experts_counter_tensor.buffer()->address();
            reader_runtime_args.at(6) = (uint32_t)shared_variables.cross_device_semaphore.address();
            reader_runtime_args.at(7) = (uint32_t)shared_variables.init_semaphore.address();

            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            writer_runtime_args.at(3) = tensor_args.chip_to_n_routed_expert_offset_tensor.buffer()->address();
            writer_runtime_args.at(4) = combined_output_tensor.buffer()->address();
            writer_runtime_args.at(5) = experts_counter_tensor.buffer()->address();
            writer_runtime_args.at(6) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(7) = (uint32_t)shared_variables.init_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch_combined

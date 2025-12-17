// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <limits>

namespace ttnn::operations::experimental::ccl {

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

}  // namespace detail

AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::cached_mesh_workload_t
AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    // Default worker core range - single core for now
    CoreRangeSet worker_core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

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

ttnn::device_operation::CachedProgram<AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::shared_variables_t>
AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::Program program{};

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const auto dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto input_scores_tensor = tensor_args.expert_scores_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;

    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);
    const auto& output_scores_tensor = tensor_return_value.at(2);

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ::ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch selective tilize program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    const auto [neighbors, directions] =
        ::ttnn::operations::ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * dispatch_devices;

    uint32_t tokens_per_device = detail::get_num_rows(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-1];

    auto input_page_size = detail::get_page_size(input_tensor);
    auto indices_page_size = detail::get_page_size(indices_tensor);
    auto mapping_page_size = detail::get_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    auto output_pages = detail::get_num_pages(output_tensor);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());
    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_scores_tensor.dtype());

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // client interface
    uint32_t packet_header_cb_id = tt::CBIndex::c_3;
    // book-keeping buffer to avoid sending the same token multiple times
    uint32_t send_preparation_buffer_id = tt::CBIndex::c_4;
    // full scores buffer
    uint32_t scores_tensor_cb_id = tt::CBIndex::c_5;

    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    log_debug(
        tt::LogOp,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size(indices_tensor);
    log_debug(
        tt::LogOp,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    log_debug(
        tt::LogOp,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size(output_tensor);
    log_debug(
        tt::LogOp,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    uint32_t aligned_metadata_page_size = detail::get_aligned_page_size(metadata_tensor);
    log_debug(
        tt::LogOp,
        "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}, aligned_metadata_page_size: {}",
        metadata_tensor.logical_shape(),
        metadata_pages,
        metadata_page_size,
        aligned_metadata_page_size);

    constexpr uint32_t buffering_factor = 2;
    constexpr uint32_t num_packet_headers = 2;

    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    // Default worker core range - single core for now
    CoreRangeSet worker_core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    uint32_t num_cores = worker_core_range_set.num_cores();
    uint32_t subtoken_bytes_aligned = tt::align(tt::div_up(aligned_input_page_size, num_cores), dram_alignment);
    uint32_t subtoken_units_of_work = tt::div_up(aligned_input_page_size, subtoken_bytes_aligned);
    uint32_t max_indices_pages_per_packet = tt::div_up(fabric_max_packet_size, aligned_indices_page_size);

    // split each token of H=7168 (hidden size) into subtoken_units_of_work subtokens, and have each core send part of
    // the token
    auto
        [num_tokens_cores,
         all_tokens_cores,
         subtoken_cores_group_1,
         subtoken_cores_group_2,
         subtoken_units_per_core_g1,
         subtoken_units_per_core_g2] = tt::tt_metal::split_work_to_cores(worker_core_range_set, subtoken_units_of_work);

    auto
        [num_indices_cores,
         all_indices_cores,
         indices_cores_group_1,
         indices_cores_group_2,
         indices_units_per_core_g1,
         indices_units_per_core_g2] = tt::tt_metal::split_work_to_cores(worker_core_range_set, indices_pages);

    uint32_t max_subtoken_size =
        std::max(subtoken_units_per_core_g1, subtoken_units_per_core_g2) * subtoken_bytes_aligned;

    auto sender_core_grid = all_tokens_cores.merge(all_indices_cores);

    auto sender_cores = corerange_to_cores(sender_core_grid);

    // Create circular buffers

    // Store subtokens of the input tensor in a circular buffer
    tt::tt_metal::create_cb(
        input_tensor_cb_id, program, sender_core_grid, max_subtoken_size, buffering_factor, input_data_format);

    // Store entire indices tensor in a circular buffer
    tt::tt_metal::create_cb(
        indices_tensor_cb_id,
        program,
        sender_core_grid,
        aligned_indices_page_size,
        2 * max_indices_pages_per_packet,
        indices_data_format);

    // Store entire scores tensor in a circular buffer
    tt::tt_metal::create_cb(
        scores_tensor_cb_id,
        program,
        sender_core_grid,
        aligned_indices_page_size,
        2 * max_indices_pages_per_packet,  // scores tensor is the same size as the indices tensor
        scores_data_format);

    // Store entire mapping tensor in a circular buffer
    tt::tt_metal::create_cb(
        mapping_tensor_cb_id, program, sender_core_grid, aligned_mapping_page_size, mapping_pages, mapping_data_format);

    // Store send preparation buffer in a circular buffer
    tt::tt_metal::create_cb(
        send_preparation_buffer_id,
        program,
        sender_core_grid,
        tokens_per_device * sizeof(uint8_t),
        num_devices,
        tt::DataFormat::UInt8);

    // Store packet header buffer in a circular buffer
    tt::tt_metal::create_cb(
        packet_header_cb_id,
        program,
        sender_core_grid,
        packet_header_size_bytes,
        num_packet_headers,
        tt::DataFormat::RawUInt32);

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", ::ttnn::operations::ccl::common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", ::ttnn::operations::ccl::common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", ::ttnn::operations::ccl::common::stringify(directions));

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"input_tensor_cb_id", input_tensor_cb_id},
        {"indices_tensor_cb_id", indices_tensor_cb_id},
        {"mapping_tensor_cb_id", mapping_tensor_cb_id},
        {"packet_header_cb_id", packet_header_cb_id},
        {"send_preparation_buffer_id", send_preparation_buffer_id},

        {"input_pages", input_pages},
        {"indices_pages", indices_pages},
        {"mapping_pages", mapping_pages},
        {"output_pages", output_pages},
        {"metadata_pages", metadata_pages},

        {"input_page_size", input_page_size},
        {"indices_page_size", indices_page_size},
        {"mapping_page_size", mapping_page_size},
        {"output_page_size", output_page_size},
        {"metadata_page_size", metadata_page_size},

        {"num_devices", num_devices},
        {"hidden_size", hidden_size},
        {"batch_size", batch_size},
        {"selected_experts_k", selected_experts_k},
        {"experts", experts},
        {"tokens_per_device", tokens_per_device},

        {"num_links", num_links},
        {"topology", (uint32_t)topology},

        {"src_mesh_id", src_mesh_id},
        {"src_chip_id", (uint32_t)src_chip_id},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},

        {"aligned_input_page_size", aligned_input_page_size},
        {"aligned_indices_page_size", aligned_indices_page_size},
        {"aligned_mapping_page_size", aligned_mapping_page_size},
        {"aligned_output_page_size", aligned_output_page_size},
        {"aligned_metadata_page_size", aligned_metadata_page_size},

        {"fabric_max_packet_size", (uint32_t)fabric_max_packet_size},

        {"l1_alignment", l1_alignment},
        {"dram_alignment", dram_alignment},
        {"linearized_mesh_coord", linearized_mesh_coord},
    };

    std::vector<uint32_t> compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(mapping_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(compile_time_args);

    std::map<std::string, std::string> reader_defines = {};

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/dataflow/"
        "reader_all_to_all_dispatch_selective_tilize.cpp",
        sender_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, {}, named_compile_time_args, reader_defines));

    // Code-gen a mesh-position to fabric chip ID array for the writer kernel
    // Code-gen a mesh-position to mesh-id array for the writer kernel
    // Code-gen a direction array that is set to true when a direction has a valid connection (when a neighbor exists or
    // if it's along a valid cluster axis)
    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", ::ttnn::operations::ccl::common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", ::ttnn::operations::ccl::common::stringify(dest_mesh_id)},
        {"DIRECTIONS", ::ttnn::operations::ccl::common::stringify(directions)}};

    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/dataflow/"
        "writer_all_to_all_dispatch_selective_tilize.cpp",
        sender_core_grid,
        tt::tt_metal::WriterDataMovementConfig(compile_time_args, {}, named_compile_time_args, writer_defines));

    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),            // 0
        indices_tensor.buffer()->address(),          // 1
        mapping_tensor.buffer()->address(),          // 2
        output_tensor.buffer()->address(),           // 3
        metadata_tensor.buffer()->address(),         // 4
        (uint32_t)cross_device_semaphore.address(),  // 5
        (uint32_t)init_semaphore.address(),          // 6
        input_scores_tensor.buffer()->address(),     // 7
        output_scores_tensor.buffer()->address(),    // 8
    };

    uint32_t subtoken_offset_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);
    uint32_t subtoken_size_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);
    uint32_t indices_start_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);
    uint32_t indices_end_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);

    uint32_t link_id = 0;
    uint32_t subtoken_offset = 0;
    uint32_t indices_start = 0;
    for (uint32_t i = 0; i < sender_cores.size(); i++) {
        std::vector<uint32_t> writer_runtime_args = reader_runtime_args;

        if (subtoken_cores_group_1.contains(sender_cores.at(i))) {
            reader_runtime_args.at(subtoken_offset_idx) = subtoken_offset;
            uint32_t subtoken_size = subtoken_units_per_core_g1 * subtoken_bytes_aligned;
            if (subtoken_offset + subtoken_size > aligned_input_page_size) {
                subtoken_size = aligned_input_page_size - subtoken_offset;
            }
            reader_runtime_args.at(subtoken_size_idx) = subtoken_size;
            subtoken_offset += reader_runtime_args.at(subtoken_size_idx);
        } else if (subtoken_cores_group_2.contains(sender_cores.at(i))) {
            reader_runtime_args.at(subtoken_offset_idx) = subtoken_offset;
            uint32_t subtoken_size = subtoken_units_per_core_g2 * subtoken_bytes_aligned;
            if (subtoken_offset + subtoken_size > aligned_input_page_size) {
                subtoken_size = aligned_input_page_size - subtoken_offset;
            }
            reader_runtime_args.at(subtoken_size_idx) = subtoken_size;
            subtoken_offset += reader_runtime_args.at(subtoken_size_idx);
        }

        if (indices_cores_group_1.contains(sender_cores.at(i))) {
            reader_runtime_args.at(indices_start_idx) = indices_start;
            reader_runtime_args.at(indices_end_idx) =
                std::min(indices_start + indices_units_per_core_g1, indices_pages);
            indices_start = reader_runtime_args.at(indices_end_idx);
        } else if (indices_cores_group_2.contains(sender_cores.at(i))) {
            reader_runtime_args.at(indices_start_idx) = indices_start;
            reader_runtime_args.at(indices_end_idx) =
                std::min(indices_start + indices_units_per_core_g2, indices_pages);
            indices_start = reader_runtime_args.at(indices_end_idx);
        }

        for (const auto& neighbor_coordinate : neighbors) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id,
                mesh_device->get_fabric_node_id(neighbor_coordinate),
                link_id,
                program,
                sender_cores.at(i),
                writer_runtime_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, sender_cores.at(i), reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, binary_writer_kernel_id, sender_cores.at(i), writer_runtime_args);
        link_id++;
    }

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .binary_writer_kernel_id = binary_writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        const auto& binary_writer_kernel_id = shared_variables.binary_writer_kernel_id;
        const auto& cores = shared_variables.cores;

        const auto& output_tensor = tensor_return_value.at(0);
        const auto& metadata_tensor = tensor_return_value.at(1);
        const auto& output_scores_tensor = tensor_return_value.at(2);

        for (const auto& core : cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, ternary_reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, binary_writer_kernel_id, core);

            // Update reader runtime args to match create_at layout:
            // 0: input_tensor, 1: indices_tensor, 2: mapping_tensor, 3: output_tensor,
            // 4: metadata_tensor, 5: cross_device_semaphore, 6: init_semaphore,
            // 7: input_scores_tensor, 8: output_scores_tensor
            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.expert_mapping_tensor.buffer()->address();
            reader_runtime_args.at(3) = output_tensor.buffer()->address();
            reader_runtime_args.at(4) = metadata_tensor.buffer()->address();
            reader_runtime_args.at(5) = (uint32_t)shared_variables.cross_device_semaphore.address();
            reader_runtime_args.at(6) = (uint32_t)shared_variables.init_semaphore.address();
            reader_runtime_args.at(7) = tensor_args.expert_scores_tensor.buffer()->address();
            reader_runtime_args.at(8) = output_scores_tensor.buffer()->address();

            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.expert_mapping_tensor.buffer()->address();
            writer_runtime_args.at(3) = output_tensor.buffer()->address();
            writer_runtime_args.at(4) = metadata_tensor.buffer()->address();
            writer_runtime_args.at(5) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(6) = (uint32_t)shared_variables.init_semaphore.address();
            writer_runtime_args.at(7) = tensor_args.expert_scores_tensor.buffer()->address();
            writer_runtime_args.at(8) = output_scores_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl

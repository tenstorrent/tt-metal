// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
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

uint32_t get_num_pages_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows_st(const ttnn::Tensor& tensor) {
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

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<
    AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::shared_variables_t>
AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::Program program{};

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const auto dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto input_scores_tensor = tensor_args.expert_scores_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;

    const auto& output_tensor = tensor_return_value;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    uint32_t linearized_mesh_coord = ::ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "Creating selective tilize program for mesh coordinate: ({}, {}) linearized: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        linearized_mesh_coord);

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_per_device = input_shape[0];
    uint32_t batch_size = input_shape[0] * dispatch_devices;

    uint32_t tokens = detail::get_num_rows_st(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-1];
    uint32_t experts_per_device = tt::div_up(experts, num_devices);

    auto input_page_size = detail::get_page_size_st(input_tensor);
    auto indices_page_size = detail::get_page_size_st(indices_tensor);
    auto mapping_page_size = detail::get_page_size_st(mapping_tensor);
    auto output_page_size = detail::get_page_size_st(output_tensor);

    auto input_pages = detail::get_num_pages_st(input_tensor);
    auto indices_pages = detail::get_num_pages_st(indices_tensor);
    auto mapping_pages = detail::get_num_pages_st(mapping_tensor);
    auto scores_pages = detail::get_num_pages_st(input_scores_tensor);
    auto output_pages = detail::get_num_pages_st(output_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());
    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_scores_tensor.dtype());

    // CB for passing total_chunks from writer to compute
    uint32_t total_chunks_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // full scores buffer
    uint32_t scores_tensor_cb_id = tt::CBIndex::c_3;
    // Send preparation buffer [E, T] for untilize, capped by -1 to indicate no more tokens to send for this expert
    uint32_t e_t_buffer_id = tt::CBIndex::c_4;
    // Tilizer input buffer for tokens to be tilized (row-major from reader)
    uint32_t tilizer_input_cb_id = tt::CBIndex::c_5;
    // Tilizer output buffer for tilized tokens (from compute to writer)
    uint32_t tilizer_output_cb_id = tt::CBIndex::c_6;
    // Experts activation buffer [T, E + 1] {token id, expert_0_activated, expert_1_activated, ...}
    // k+1 if not activated, k value in the indices tensor for that token if activated
    uint32_t expert_activation_cb_id = tt::CBIndex::c_7;
    // after determining the total number of tokens for each expert, this buffer will store the total number of tokens
    // for each expert to pass to the other kernels
    uint32_t per_expert_total_tokens_cb_id = tt::CBIndex::c_8;

    uint32_t aligned_input_page_size = detail::get_aligned_page_size_st(input_tensor);
    log_debug(
        tt::LogOp,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size_st(indices_tensor);
    log_debug(
        tt::LogOp,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size_st(mapping_tensor);
    log_debug(
        tt::LogOp,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size_st(output_tensor);
    log_debug(
        tt::LogOp,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    CoreRangeSet selective_tilize_core_range_set = operation_attributes.selective_tilize_core_range_set.value();

    uint32_t num_cores = selective_tilize_core_range_set.num_cores();

    // Split token subregions across tilizer cores (similar to sender subtoken splitting)
    // Each tilizer core handles a portion of the hidden dimension for each token
    uint32_t tilizer_subtoken_bytes_aligned = tt::align(tt::div_up(aligned_input_page_size, num_cores), l1_alignment);
    uint32_t tilizer_subtoken_units_of_work = tt::div_up(aligned_input_page_size, tilizer_subtoken_bytes_aligned);

    auto
        [num_tilizer_work_cores,
         all_tilizer_work_cores,
         tilizer_cores_group_1,
         tilizer_cores_group_2,
         tilizer_units_per_core_g1,
         tilizer_units_per_core_g2] =
            tt::tt_metal::split_work_to_cores(selective_tilize_core_range_set, tilizer_subtoken_units_of_work);

    uint32_t max_tilizer_subtoken_size =
        std::max(tilizer_units_per_core_g1, tilizer_units_per_core_g2) * tilizer_subtoken_bytes_aligned;

    uint32_t num_tilizer_cores = selective_tilize_core_range_set.num_cores();
    auto selective_tilize_cores = corerange_to_cores(selective_tilize_core_range_set, std::nullopt, true);
    constexpr uint32_t buffering_factor = 2;
    uint32_t tokens_per_chunk = operation_attributes.tokens_per_chunk;

    // Create semaphores for synchronizing the drain tilizer to non-drain tilizers
    auto e_t_buffer_ready_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, selective_tilize_core_range_set, INVALID);
    // Semaphore for drain tilizer to signal non-drain tilizers that E-D table computation is complete
    // auto combine_and_matmul_core_range_set =
    // operation_attributes.matmul_core_range_set.merge(operation_attributes.combine_core_range_set)

    // Get the bounding box of tilizer cores for multicast from drain tilizer to non-drain tilizers
    // Used for E-D table computed signal
    auto tilizer_bbox = selective_tilize_core_range_set.bounding_box();
    CoreCoord tilizer_mcast_start_logical = tilizer_bbox.start_coord;
    CoreCoord tilizer_mcast_end_logical = tilizer_bbox.end_coord;

    auto matmul_bbox = operation_attributes.matmul_core_range_set.value().bounding_box();
    CoreCoord matmul_mcast_start_logical = matmul_bbox.start_coord;
    CoreCoord matmul_mcast_end_logical = matmul_bbox.end_coord;

    auto combine_bbox = operation_attributes.combine_core_range_set.value().bounding_box();
    CoreCoord combine_mcast_start_logical = combine_bbox.start_coord;
    CoreCoord combine_mcast_end_logical = combine_bbox.end_coord;

    // Convert to physical NOC coordinates
    auto tilizer_mcast_start_physical = mesh_device->worker_core_from_logical_core(tilizer_mcast_start_logical);
    auto tilizer_mcast_end_physical = mesh_device->worker_core_from_logical_core(tilizer_mcast_end_logical);
    auto matmul_mcast_start_physical = mesh_device->worker_core_from_logical_core(matmul_mcast_start_logical);
    auto matmul_mcast_end_physical = mesh_device->worker_core_from_logical_core(matmul_mcast_end_logical);
    auto combine_mcast_start_physical = mesh_device->worker_core_from_logical_core(combine_mcast_start_logical);
    auto combine_mcast_end_physical = mesh_device->worker_core_from_logical_core(combine_mcast_end_logical);

    // For NOC 0: start = (min_x, min_y), end = (max_x, max_y)
    // For NOC 1: coordinates are swapped
    // We'll use NOC 0 by default, but pass both orderings and let the kernel handle it
    // Or we can determine the NOC here and swap if needed
    // For simplicity, we pass the NOC 0 ordering (start < end) and the kernel will use NOC 0

    tt::tt_metal::create_cb(
        e_t_buffer_id,
        program,
        selective_tilize_core_range_set,
        tokens * sizeof(uint32_t),  // total tokens * sizeof(uint32_t)
        experts_per_device,         // number of experts on the device
        tt::DataFormat::UInt32);

    // Assume indices tensor is sharded in L1
    tt::tt_metal::create_cb(
        indices_tensor_cb_id,
        program,
        selective_tilize_core_range_set,
        aligned_indices_page_size,
        indices_pages,  // double buffer buffer packets
        indices_data_format,
        indices_tensor.buffer());

    // Assume scores tensor is sharded in L1
    tt::tt_metal::create_cb(
        scores_tensor_cb_id,
        program,
        selective_tilize_core_range_set,
        aligned_indices_page_size,
        scores_pages,
        scores_data_format,
        input_scores_tensor.buffer());

    // For each batch's tokens, we need to read the relevant experts from the mapping tensor
    // For in range (tokens) every time tokens/batch increments, read in new mapping tensor page
    tt::tt_metal::create_cb(
        mapping_tensor_cb_id,
        program,
        selective_tilize_core_range_set,
        aligned_mapping_page_size,
        2,
        mapping_data_format);

    // Tilizer input buffer: holds subtokens for tokens_per_chunk tokens, double-buffered
    // Each tilizer core reads its subtoken portion of incoming tokens
    tt::tt_metal::create_cb(
        tilizer_input_cb_id,
        program,
        selective_tilize_core_range_set,
        max_tilizer_subtoken_size,
        operation_attributes.tokens_per_chunk * buffering_factor,  // double-buffered tokens_per_chunk
        input_data_format);

    tt::tt_metal::create_cb(
        expert_activation_cb_id,
        program,
        selective_tilize_core_range_set,
        selected_experts_k * sizeof(uint32_t),
        tokens,
        tt::DataFormat::UInt32);

    tt::tt_metal::create_cb(
        per_expert_total_tokens_cb_id,
        program,
        selective_tilize_core_range_set,
        sizeof(uint32_t),  // at most 512 for decode
        experts_per_device,
        tt::DataFormat::UInt32);

    // CB for passing total_chunks from writer to compute kernel
    // Single page holding one uint32_t value
    tt::tt_metal::create_cb(
        total_chunks_cb_id,
        program,
        selective_tilize_core_range_set,
        sizeof(uint32_t),
        1,  // single page
        tt::DataFormat::UInt32);

    // Tilizer output buffer: holds tilized output from compute kernel
    // page_size is the tile size, num_pages is tiles_per_chunk (based on max subtoken size)
    // Tile dimensions: height = tokens_per_chunk, width = 32
    // tile_width_bytes = TILE_WIDTH * element_size
    // tiles_per_chunk = max_tilizer_subtoken_size / tile_width_bytes
    constexpr uint32_t TILE_WIDTH = 32;
    // uint32_t element_size = input_tensor.element_size();
    // uint32_t bfp8_tile_size = 1088 * sizeof(uint8_t);
    uint32_t tile_width_bytes = TILE_WIDTH * input_tensor.element_size();
    uint32_t tiles_per_chunk = max_tilizer_subtoken_size / tile_width_bytes;
    tt::tt_metal::create_cb(
        tilizer_output_cb_id,
        program,
        selective_tilize_core_range_set,
        tokens_per_chunk * tile_width_bytes,
        tiles_per_chunk * buffering_factor,  // double-buffered
        input_data_format);

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"tilizer_input_cb_id", tilizer_input_cb_id},
        {"tilizer_output_cb_id", tilizer_output_cb_id},
        {"total_chunks_cb_id", total_chunks_cb_id},
        {"indices_tensor_cb_id", indices_tensor_cb_id},
        {"scores_tensor_cb_id", scores_tensor_cb_id},
        {"mapping_tensor_cb_id", mapping_tensor_cb_id},
        {"e_t_buffer_id", e_t_buffer_id},
        {"expert_activation_cb_id", expert_activation_cb_id},
        {"per_expert_total_tokens_cb_id", per_expert_total_tokens_cb_id},
        {"input_pages", input_pages},
        {"indices_pages", indices_pages},
        {"mapping_pages", mapping_pages},
        {"scores_pages", scores_pages},
        {"output_pages", output_pages},

        {"input_page_size", input_page_size},
        {"indices_page_size", indices_page_size},
        {"mapping_page_size", mapping_page_size},
        {"output_page_size", output_page_size},

        {"num_devices", num_devices},
        {"hidden_size", hidden_size},
        {"batch_size", batch_size},
        {"batch_per_device", batch_per_device},
        {"selected_experts_k", selected_experts_k},
        {"experts", experts},
        {"tokens", tokens},

        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},

        {"aligned_input_page_size", aligned_input_page_size},
        {"aligned_indices_page_size", aligned_indices_page_size},
        {"aligned_mapping_page_size", aligned_mapping_page_size},
        {"aligned_output_page_size", aligned_output_page_size},

        {"l1_alignment", l1_alignment},
        {"dram_alignment", dram_alignment},
        {"linearized_mesh_coord", linearized_mesh_coord},
        {"cluster_axis", (uint32_t)operation_attributes.axis.value()},

        // Multicast coordinates for drain tilizer to non-drain tilizer synchronization
        {"tilizer_mcast_start_x", (uint32_t)tilizer_mcast_start_physical.x},
        {"tilizer_mcast_start_y", (uint32_t)tilizer_mcast_start_physical.y},
        {"tilizer_mcast_end_x", (uint32_t)tilizer_mcast_end_physical.x},
        {"tilizer_mcast_end_y", (uint32_t)tilizer_mcast_end_physical.y},
        {"matmul_mcast_start_x", (uint32_t)matmul_mcast_start_physical.x},
        {"matmul_mcast_start_y", (uint32_t)matmul_mcast_start_physical.y},
        {"matmul_mcast_end_x", (uint32_t)matmul_mcast_end_physical.x},
        {"matmul_mcast_end_y", (uint32_t)matmul_mcast_end_physical.y},
        {"combine_mcast_start_x", (uint32_t)combine_mcast_start_physical.x},
        {"combine_mcast_start_y", (uint32_t)combine_mcast_start_physical.y},
        {"combine_mcast_end_x", (uint32_t)combine_mcast_end_physical.x},
        {"combine_mcast_end_y", (uint32_t)combine_mcast_end_physical.y},
        {"num_tilizer_cores", num_tilizer_cores},
        {"num_mm_cores", operation_attributes.matmul_core_range_set.value().num_cores()},
        {"num_combine_cores", operation_attributes.combine_core_range_set.value().num_cores()},
        {"tiles_per_chunk", tiles_per_chunk},
        {"tokens_per_chunk", operation_attributes.tokens_per_chunk},
        {"e_t_buffer_ready_semaphore_id", e_t_buffer_ready_semaphore_id},
    };

    std::vector<uint32_t> compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_scores_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(mapping_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);

    tt::tt_metal::KernelHandle selective_tilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/dataflow/"
        "reader_tilizer.cpp",
        selective_tilize_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, {}, named_compile_time_args));

    tt::tt_metal::KernelHandle writer_tilizer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/dataflow/"
        "writer_tilizer.cpp",
        selective_tilize_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(compile_time_args, {}, named_compile_time_args));

    // Compute kernel compile-time args for tilization
    // These are positional args: tilizer_input_cb_id, tilizer_output_cb_id, tiles_per_chunk,
    //   tokens_per_chunk, total_chunks_cb_id
    std::unordered_map<std::string, uint32_t> compute_tilizer_named_compile_time_args = {
        {"tilizer_input_cb_id", tilizer_input_cb_id},
        {"tilizer_output_cb_id", tilizer_output_cb_id},
        {"tiles_per_chunk", tiles_per_chunk},
        {"tokens_per_chunk", operation_attributes.tokens_per_chunk},
        {"total_chunks_cb_id", total_chunks_cb_id},
    };

    tt::tt_metal::KernelHandle compute_tilizer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/compute/"
        "compute_tilizer.cpp",
        selective_tilize_core_range_set,
        tt::tt_metal::ComputeConfig{.named_compile_args = compute_tilizer_named_compile_time_args});

    std::vector<uint32_t> selective_tilize_runtime_args = {
        input_tensor.buffer()->address(),         // 0
        indices_tensor.buffer()->address(),       // 1
        input_scores_tensor.buffer()->address(),  // 2
        mapping_tensor.buffer()->address(),       // 3
    };

    [[maybe_unused]] uint32_t is_drain_tilizer_core_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 4: is_drain_tilizer_core

    // Add work split runtime args for tilizer cores
    uint32_t tilizer_subtoken_offset_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 5: tilizer_subtoken_offset
    uint32_t tilizer_subtoken_size_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 6: tilizer_subtoken_size

    std::vector<CoreCoord> drain_tilizer_cores;
    uint32_t tilizer_subtoken_offset = 0;

    for (uint32_t i = 0; i < num_tilizer_cores; i++) {
        // Set work split parameters based on which group the core is in
        if (tilizer_cores_group_1.contains(selective_tilize_cores.at(i))) {
            selective_tilize_runtime_args.at(tilizer_subtoken_offset_idx) = tilizer_subtoken_offset;
            uint32_t tilizer_subtoken_size = tilizer_units_per_core_g1 * tilizer_subtoken_bytes_aligned;
            // Clamp to not exceed the total token size
            if (tilizer_subtoken_offset + tilizer_subtoken_size > aligned_input_page_size) {
                tilizer_subtoken_size = aligned_input_page_size - tilizer_subtoken_offset;
            }
            selective_tilize_runtime_args.at(tilizer_subtoken_size_idx) = tilizer_subtoken_size;
            tilizer_subtoken_offset += tilizer_subtoken_size;
        } else if (tilizer_cores_group_2.contains(selective_tilize_cores.at(i))) {
            selective_tilize_runtime_args.at(tilizer_subtoken_offset_idx) = tilizer_subtoken_offset;
            uint32_t tilizer_subtoken_size = tilizer_units_per_core_g2 * tilizer_subtoken_bytes_aligned;
            // Clamp to not exceed the total token size
            if (tilizer_subtoken_offset + tilizer_subtoken_size > aligned_input_page_size) {
                tilizer_subtoken_size = aligned_input_page_size - tilizer_subtoken_offset;
            }
            selective_tilize_runtime_args.at(tilizer_subtoken_size_idx) = tilizer_subtoken_size;
            tilizer_subtoken_offset += tilizer_subtoken_size;
        }

        tt::tt_metal::SetRuntimeArgs(
            program, selective_tilize_kernel_id, selective_tilize_cores.at(i), selective_tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(
            program, writer_tilizer_kernel_id, selective_tilize_cores.at(i), selective_tilize_runtime_args);
    }

    return {
        std::move(program),
        {
            .selective_tilize_kernel_id = selective_tilize_kernel_id,
            .writer_tilizer_kernel_id = writer_tilizer_kernel_id,
            .compute_tilizer_kernel_id = compute_tilizer_kernel_id,
            .selective_tilize_cores = selective_tilize_cores,
        }};
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& selective_tilize_kernel_id = shared_variables.selective_tilize_kernel_id;

        for (const auto& core : shared_variables.selective_tilize_cores) {
            auto& selective_tilize_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, selective_tilize_kernel_id, core);
            selective_tilize_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            selective_tilize_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            selective_tilize_runtime_args.at(2) = tensor_args.expert_scores_tensor.buffer()->address();
            selective_tilize_runtime_args.at(3) = tensor_args.expert_mapping_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl

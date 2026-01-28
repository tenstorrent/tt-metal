// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
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

// T, MM, T + MM
std::tuple<
    std::vector<CoreCoord>,  // T cores
    std::vector<CoreCoord>,  // MM cores
    CoreRangeSet,            // T CoreRangeSet
    CoreRangeSet,            // MM CoreRangeSet
    CoreRangeSet,            // T + MM CoreRangeSet
    CoreRange,               // T bounding box
    CoreRange>               // MM bounding box
get_cores(MeshDevice* mesh_device) {
    const std::vector<CoreCoord> t_cores = {CoreCoord(5, 0), CoreCoord(5, 1), CoreCoord(5, 2), CoreCoord(5, 3)};
    const std::vector<CoreCoord> mm_cores =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);

    const CoreRangeSet t_core_range_set = CoreRangeSet(t_cores);
    const CoreRangeSet mm_core_range_set = CoreRangeSet(mm_cores);
    const CoreRangeSet t_mm_core_range_set = t_core_range_set.merge(mm_core_range_set);

    const CoreRange t_bounding_box = t_core_range_set.bounding_box();
    const CoreRange mm_bounding_box = mm_core_range_set.bounding_box();

    TT_FATAL(!t_bounding_box.intersects(mm_bounding_box), "tilize and matmul bounding boxes cannot overlap");

    return {
        t_cores,
        mm_cores,
        t_core_range_set,
        mm_core_range_set,
        t_mm_core_range_set,
        t_bounding_box,
        mm_bounding_box,
    };
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
    [[maybe_unused]] const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::Program program{};

    // Alignment
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const auto dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // Input tensors
    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto input_scores_tensor = tensor_args.expert_scores_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    // Output tensors
    const auto& output_tensor = tensor_return_value[0];
    const auto& expert_activation_output_tensor = tensor_return_value[1];
    const auto& e_t_output_tensor = tensor_return_value[2];

    // Mesh
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t num_devices = mesh_view.num_devices();
    uint32_t linearized_mesh_coord = ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
    log_debug(
        tt::LogOp,
        "Creating selective tilize program for mesh coordinate: ({}, {}) linearized: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        linearized_mesh_coord);

    // General info
    uint32_t tokens = detail::get_num_rows_st(input_tensor);
    uint32_t hidden_size = input_shape[-1];
    uint32_t experts = mapping_shape[-1];
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts_per_device = tt::div_up(experts, num_devices);

    // Number of pages
    auto input_pages = detail::get_num_pages_st(input_tensor);
    auto indices_pages = detail::get_num_pages_st(indices_tensor);
    auto mapping_pages = detail::get_num_pages_st(mapping_tensor);
    auto scores_pages = detail::get_num_pages_st(input_scores_tensor);
    auto output_pages = detail::get_num_pages_st(output_tensor);

    // Page sizes
    auto input_page_size = detail::get_page_size_st(input_tensor);
    auto indices_page_size = detail::get_page_size_st(indices_tensor);
    auto mapping_page_size = detail::get_page_size_st(mapping_tensor);
    auto scores_page_size = detail::get_page_size_st(input_scores_tensor);
    auto output_page_size = detail::get_page_size_st(output_tensor);
    auto expert_activation_output_page_size = detail::get_page_size_st(expert_activation_output_tensor);
    auto e_t_output_page_size = detail::get_page_size_st(e_t_output_tensor);

    // Data formats
    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());
    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_scores_tensor.dtype());

    // CB indices

    // CB for passing total_chunks from writer to compute
    uint32_t total_chunks_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // full scores buffer
    uint32_t scores_tensor_cb_id = tt::CBIndex::c_3;
    // Send preparation buffer [E, T] for untilize, capped by -1 to indicate no more tokens to send for this expert
    uint32_t e_t_cb_id = tt::CBIndex::c_4;
    // Tilizer input buffer for tokens to be tilized (row-major from reader)
    uint32_t tilizer_input_cb_id = tt::CBIndex::c_5;
    // Tilizer output buffer for tilized tokens (from compute to writer)
    uint32_t tilizer_output_cb_id = tt::CBIndex::c_6;
    // Experts activation buffer [T, 2*E + 1] each row is {token id, expert_0_activated, expert_1_activated,...,
    // expert_0_score, expert_1_score, ...} k+1 if not activated, k value in the indices tensor for that token if
    // activated
    uint32_t expert_activation_cb_id = tt::CBIndex::c_7;
    // after determining the total number of tokens for each expert, this buffer will store the total number of tokens
    // for each expert to pass to the other kernels
    uint32_t per_expert_total_tokens_cb_id = tt::CBIndex::c_8;
    // BRISC's e_t buffer for parallel metadata processing (BRISC processes tokens/2 to tokens)
    uint32_t brisc_e_t_cb_id = tt::CBIndex::c_9;
    // BRISC's per-expert token counts to communicate to NCRISC after parallel processing
    uint32_t brisc_expert_counts_cb_id = tt::CBIndex::c_10;
    // BRISC's expert activation buffer for parallel processing
    uint32_t brisc_expert_activation_cb_id = tt::CBIndex::c_11;
    // BRISC's activated token count (single uint32_t)
    uint32_t brisc_activated_count_cb_id = tt::CBIndex::c_12;

    // Aligned page sizes
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

    uint32_t aligned_scores_page_size = detail::get_aligned_page_size_st(input_scores_tensor);
    log_debug(
        tt::LogOp,
        "scores shape: {}, scores_pages: {}, scores_page_size: {}, aligned_scores_page_size: {}",
        input_scores_tensor.logical_shape(),
        scores_pages,
        scores_page_size,
        aligned_scores_page_size);

    ////// CORES //////

    // Get cores to use for tilize and matmul
    const auto
        [t_cores, mm_cores, t_core_range_set, mm_core_range_set, t_mm_core_range_set, t_bounding_box, mm_bounding_box] =
            detail::get_cores(mesh_device);

    const uint32_t t_num_cores = t_core_range_set.num_cores();
    const uint32_t mm_num_cores = mm_core_range_set.num_cores();
    const uint32_t mm_bounding_box_num_cores = mm_bounding_box.size();

    // Logical mcast bounding box coordinates
    const CoreCoord t_mcast_start_logical = t_bounding_box.start_coord;
    const CoreCoord t_mcast_end_logical = t_bounding_box.end_coord;
    const CoreCoord mm_mcast_start_logical = mm_bounding_box.start_coord;
    const CoreCoord mm_mcast_end_logical = mm_bounding_box.end_coord;

    // Convert to physical NOC coordinates
    const CoreCoord t_mcast_start_physical = mesh_device->worker_core_from_logical_core(t_mcast_start_logical);
    const CoreCoord t_mcast_end_physical = mesh_device->worker_core_from_logical_core(t_mcast_end_logical);
    const CoreCoord mm_mcast_start_physical = mesh_device->worker_core_from_logical_core(mm_mcast_start_logical);
    const CoreCoord mm_mcast_end_physical = mesh_device->worker_core_from_logical_core(mm_mcast_end_logical);

    ////// WORK SPLIT //////

    // Split token subregions across tilizer cores (similar to sender subtoken splitting)
    // Each tilizer core handles a portion of the hidden dimension for each token
    uint32_t tilizer_subtoken_bytes_aligned = tt::align(tt::div_up(aligned_input_page_size, t_num_cores), l1_alignment);
    uint32_t tilizer_subtoken_units_of_work = tt::div_up(aligned_input_page_size, tilizer_subtoken_bytes_aligned);

    auto
        [num_tilizer_work_cores,
         all_tilizer_work_cores,
         tilizer_cores_group_1,
         tilizer_cores_group_2,
         tilizer_units_per_core_g1,
         tilizer_units_per_core_g2] =
            tt::tt_metal::split_work_to_cores(t_core_range_set, tilizer_subtoken_units_of_work);

    ////// SEMAPHORES //////

    // T:
    // - Signal from non-drain-sync cores to drain-sync core that partial metadata results are ready
    // MM:
    // - N/A
    auto partial_metadata_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, t_core_range_set, INVALID);

    // T:
    // - Signal from drain-sync core to non-drain-sync cores that final metadata results are ready
    // MM:
    // - T drain-sync core signals to MM cores that final metadata results are ready
    auto metadata_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, t_mm_core_range_set, INVALID);

    // T:
    // - MMs signal to Ts that space for new chunk is free
    // MM:
    // - N/A
    auto matmul_chunk_available_semaphore_id = tt::tt_metal::CreateSemaphore(program, t_mm_core_range_set, INVALID);

    // T:
    // - Signal from non-drain-sync cores to drain-sync core that partial chunk has been sent to MM cores
    // MM:
    // - N/A
    // NOTE:
    // - Need a semaphore per expert to avoid a race condition where each expert has a single chunk
    // - Removes requirement that non-drain-sync cores have to wait to signal until drain-sync core
    //   signals that the previous matmul_chunk_ready_semaphore has been multicasted
    constexpr uint32_t supported_num_experts_per_device = 2;
    TT_FATAL(
        experts_per_device <= supported_num_experts_per_device,
        "requires a semaphore per expert, expected {} experts per device but got {}",
        supported_num_experts_per_device,
        experts_per_device);
    auto e0_tilize_chunk_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, t_core_range_set, INVALID);
    auto e1_tilize_chunk_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, t_core_range_set, INVALID);

    // T:
    // - N/A
    // MM:
    // - T drain-sync signal to MMs that chunk has arrived
    auto matmul_chunk_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, t_mm_core_range_set, INVALID);

    ////// CBs //////

    uint32_t max_tilizer_subtoken_size =
        std::max(tilizer_units_per_core_g1, tilizer_units_per_core_g2) * tilizer_subtoken_bytes_aligned;

    constexpr uint32_t buffering_factor = 2;
    uint32_t tokens_per_chunk = operation_attributes.tokens_per_chunk;

    // e_t buffer entry size must be 16B aligned for NOC DMA during BRISC->NCRISC merge
    // 16B per token entry for NOC alignment
    constexpr uint32_t e_t_entry_size = 16;

    tt::tt_metal::create_cb(
        e_t_cb_id,
        program,
        t_core_range_set,
        e_t_output_page_size,
        experts_per_device,  // number of experts on the device
        tt::DataFormat::UInt32);

    // Assume indices tensor is sharded in L1
    tt::tt_metal::create_cb(
        indices_tensor_cb_id,
        program,
        t_core_range_set,
        aligned_indices_page_size,
        indices_pages,  // double buffer buffer packets
        indices_data_format,
        indices_tensor.buffer());

    // Assume scores tensor is sharded in L1
    tt::tt_metal::create_cb(
        scores_tensor_cb_id,
        program,
        t_core_range_set,
        aligned_indices_page_size,
        scores_pages,
        scores_data_format,
        input_scores_tensor.buffer());

    // For each batch's tokens, we need to read the relevant experts from the mapping tensor
    // For in range (tokens) every time tokens/batch increments, read in new mapping tensor page
    tt::tt_metal::create_cb(
        mapping_tensor_cb_id, program, t_core_range_set, aligned_mapping_page_size, mapping_pages, mapping_data_format);

    // Tilizer input buffer: holds subtokens for tokens_per_chunk tokens, double-buffered
    // Each tilizer core reads its subtoken portion of incoming tokens
    tt::tt_metal::create_cb(
        tilizer_input_cb_id,
        program,
        t_core_range_set,
        max_tilizer_subtoken_size,
        tokens_per_chunk * buffering_factor,  // double-buffered tokens_per_chunk
        input_data_format);

    tt::tt_metal::create_cb(
        expert_activation_cb_id,
        program,
        t_core_range_set,
        tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment),
        tokens,
        tt::DataFormat::UInt32);

    tt::tt_metal::create_cb(
        per_expert_total_tokens_cb_id,
        program,
        t_mm_core_range_set,  // allocated on T and MM cores
        sizeof(uint32_t),     // at most the value "512" for decode
        experts_per_device,
        tt::DataFormat::UInt32);

    // BRISC's e_t buffer for parallel metadata processing
    // BRISC processes the second half of tokens (tokens/2 to tokens)
    // Single page containing all experts' token lists, each with capacity tokens/2
    // Uses same 16B entry alignment as main e_t buffer for NOC DMA compatibility
    tt::tt_metal::create_cb(
        brisc_e_t_cb_id,
        program,
        t_core_range_set,
        (tokens / 2) * e_t_entry_size * experts_per_device,  // full buffer with 16B entries
        1,
        tt::DataFormat::UInt32);

    // BRISC's per-expert token counts to communicate to NCRISC
    // Single page containing counts for all experts
    tt::tt_metal::create_cb(
        brisc_expert_counts_cb_id,
        program,
        t_core_range_set,
        sizeof(uint32_t) * experts_per_device,  // all counts in one page
        1,
        tt::DataFormat::UInt32);

    // BRISC's expert activation buffer - same format as main expert_activation buffer
    // [token_id, k_indices[experts_per_device], scores[experts_per_device]] per activated token
    // Single page containing all activation rows (max tokens/2)
    uint32_t brisc_activation_row_size = tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment);
    tt::tt_metal::create_cb(
        brisc_expert_activation_cb_id,
        program,
        t_core_range_set,
        brisc_activation_row_size * (tokens / 2),  // full buffer in one page
        1,
        tt::DataFormat::UInt32);

    // BRISC's activated token count (single uint32_t to communicate to NCRISC)
    tt::tt_metal::create_cb(
        brisc_activated_count_cb_id, program, t_core_range_set, sizeof(uint32_t), 1, tt::DataFormat::UInt32);

    // CB for receiving counts from non-drain tilizer cores (only used on drain core)
    // Each non-drain core sends: [e_t_count_expert0, e_t_count_expert1, activated_count]
    // Layout: 3 values per core × (t_num_cores - 1) cores, 16B aligned per core's data
    uint32_t remote_counts_cb_id = tt::CBIndex::c_13;
    uint32_t counts_per_remote_core = experts_per_device + 1;  // e_t counts + activated count
    uint32_t remote_counts_entry_size = tt::align(counts_per_remote_core * sizeof(uint32_t), l1_alignment);
    tt::tt_metal::create_cb(
        remote_counts_cb_id,
        program,
        t_core_range_set,
        remote_counts_entry_size,
        t_num_cores - 1,  // one entry per non-drain core
        tt::DataFormat::UInt32);

    // CB for passing total_chunks from writer to compute kernel
    // Single page holding one uint32_t value
    tt::tt_metal::create_cb(
        total_chunks_cb_id,
        program,
        t_core_range_set,
        sizeof(uint32_t),
        1,  // single page
        tt::DataFormat::UInt32);

    // Tilizer output buffer: holds tilized output from compute kernel
    // page_size is the tile size, num_pages is max_tiles_per_chunk (based on max subtoken size)
    // Tile dimensions: height = tokens_per_chunk, width = 32
    // tile_width_bytes = TILE_WIDTH * element_size
    // max_tiles_per_chunk = max_tilizer_subtoken_size / tile_width_bytes
    constexpr uint32_t TILE_WIDTH = 32;
    uint32_t tile_width_bytes = TILE_WIDTH * input_tensor.element_size();
    uint32_t max_tiles_per_chunk = max_tilizer_subtoken_size / tile_width_bytes;
    tt::tt_metal::create_cb(
        tilizer_output_cb_id,
        program,
        t_core_range_set,
        tokens_per_chunk * tile_width_bytes,
        max_tiles_per_chunk * buffering_factor,  // double-buffered
        input_data_format);

    // For NOC 0: start = (min_x, min_y), end = (max_x, max_y)
    // For NOC 1: coordinates are swapped
    // We'll use NOC 0 by default, but pass both orderings and let the kernel handle it
    // Or we can determine the NOC here and swap if needed
    // For simplicity, we pass the NOC 0 ordering (start < end) and the kernel will use NOC 0

    // Store physical NOC coordinates of all tilizer cores for cross-core communication
    // Used by drain core to read from non-drain cores, and non-drain to write to drain
    std::vector<CoreCoord> tilizer_cores_physical;
    for (uint32_t i = 0; i < t_num_cores; i++) {
        tilizer_cores_physical.push_back(mesh_device->worker_core_from_logical_core(t_cores.at(i)));
    }

    // Drain core is always the first tilizer core (index 0)
    CoreCoord drain_core_physical = tilizer_cores_physical.at(0);

    std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        // CBs
        {"tilizer_input_cb_id", tilizer_input_cb_id},
        {"tilizer_output_cb_id", tilizer_output_cb_id},
        {"total_chunks_cb_id", total_chunks_cb_id},
        {"indices_tensor_cb_id", indices_tensor_cb_id},
        {"scores_tensor_cb_id", scores_tensor_cb_id},
        {"mapping_tensor_cb_id", mapping_tensor_cb_id},
        {"e_t_cb_id", e_t_cb_id},
        {"expert_activation_cb_id", expert_activation_cb_id},
        {"per_expert_total_tokens_cb_id", per_expert_total_tokens_cb_id},
        {"brisc_e_t_cb_id", brisc_e_t_cb_id},
        {"brisc_expert_counts_cb_id", brisc_expert_counts_cb_id},
        {"brisc_expert_activation_cb_id", brisc_expert_activation_cb_id},
        {"brisc_activated_count_cb_id", brisc_activated_count_cb_id},
        {"remote_counts_cb_id", remote_counts_cb_id},

        // Alignment
        {"l1_alignment", l1_alignment},
        {"dram_alignment", dram_alignment},
        {"e_t_entry_size", e_t_entry_size},

        // Number of pages
        {"input_pages", input_pages},
        {"indices_pages", indices_pages},
        {"mapping_pages", mapping_pages},
        {"scores_pages", scores_pages},
        {"output_pages", output_pages},

        // Page sizes
        {"input_page_size", input_page_size},
        {"indices_page_size", indices_page_size},
        {"mapping_page_size", mapping_page_size},
        {"output_page_size", output_page_size},
        {"expert_activation_output_page_size", expert_activation_output_page_size},
        {"e_t_output_page_size", e_t_output_page_size},

        // Aligned page sizes
        {"aligned_input_page_size", aligned_input_page_size},
        {"aligned_indices_page_size", aligned_indices_page_size},
        {"aligned_mapping_page_size", aligned_mapping_page_size},
        {"aligned_output_page_size", aligned_output_page_size},
        {"aligned_scores_page_size", aligned_scores_page_size},

        // General info
        {"tokens", tokens},
        {"hidden_size", hidden_size},
        {"remote_counts_entry_size", remote_counts_entry_size},
        {"experts", experts},
        {"selected_experts_k", selected_experts_k},

        // Chunk info
        {"tokens_per_chunk", tokens_per_chunk},

        // Mesh
        {"num_devices", num_devices},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},
        {"linearized_mesh_coord", linearized_mesh_coord},
        {"cluster_axis", (uint32_t)operation_attributes.axis.value()},

        // Multicast coordinates for drain tilizer to non-drain tilizer synchronization
        {"drain_core_noc_x", (uint32_t)drain_core_physical.x},
        {"drain_core_noc_y", (uint32_t)drain_core_physical.y},

        // T multicast coordinates
        {"tilizer_mcast_start_x", (uint32_t)t_mcast_start_physical.x},
        {"tilizer_mcast_start_y", (uint32_t)t_mcast_start_physical.y},
        {"tilizer_mcast_end_x", (uint32_t)t_mcast_end_physical.x},
        {"tilizer_mcast_end_y", (uint32_t)t_mcast_end_physical.y},
        {"num_tilizer_cores", t_num_cores},

        // MM multicast coordinates
        {"matmul_mcast_start_x", (uint32_t)mm_mcast_start_physical.x},
        {"matmul_mcast_start_y", (uint32_t)mm_mcast_start_physical.y},
        {"matmul_mcast_end_x", (uint32_t)mm_mcast_end_physical.x},
        {"matmul_mcast_end_y", (uint32_t)mm_mcast_end_physical.y},
        {"num_matmul_cores", mm_num_cores},
        {"num_matmul_bounding_box_cores", mm_bounding_box_num_cores},

        // Semaphores
        {"partial_metadata_ready_semaphore_id", partial_metadata_ready_semaphore_id},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
        {"e0_tilize_chunk_ready_semaphore_id", e0_tilize_chunk_ready_semaphore_id},
        {"e1_tilize_chunk_ready_semaphore_id", e1_tilize_chunk_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
    };

    std::vector<uint32_t> compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_scores_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(mapping_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(expert_activation_output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(e_t_output_tensor.buffer()).append_to(compile_time_args);

    tt::tt_metal::KernelHandle selective_tilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/dataflow/"
        "reader_tilizer.cpp",
        t_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, {}, named_compile_time_args));

    tt::tt_metal::KernelHandle writer_tilizer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/dataflow/"
        "writer_tilizer.cpp",
        t_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(compile_time_args, {}, named_compile_time_args));

    // Compute kernel compile-time args for tilization
    std::unordered_map<std::string, uint32_t> compute_tilizer_named_compile_time_args = {
        {"tilizer_input_cb_id", tilizer_input_cb_id},
        {"tilizer_output_cb_id", tilizer_output_cb_id},
        {"total_chunks_cb_id", total_chunks_cb_id},
        {"tokens_per_chunk", tokens_per_chunk},
        {"max_tiles_per_chunk", max_tiles_per_chunk},
    };

    tt::tt_metal::KernelHandle compute_tilizer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_selective_tilize/device/kernels/compute/"
        "compute_tilizer.cpp",
        t_core_range_set,
        tt::tt_metal::ComputeConfig{.named_compile_args = compute_tilizer_named_compile_time_args});

    std::vector<uint32_t> selective_tilize_runtime_args = {
        input_tensor.buffer()->address(),                     // 0
        indices_tensor.buffer()->address(),                   // 1
        input_scores_tensor.buffer()->address(),              // 2
        mapping_tensor.buffer()->address(),                   // 3
        output_tensor.buffer()->address(),                    // 4
        expert_activation_output_tensor.buffer()->address(),  // 5
        e_t_output_tensor.buffer()->address(),                // 6
        expert_activation_output_tensor.buffer()
            ->address(),  // 7  // TODO: (GR) placeholder for matmul_chunk_input_tensor
    };

    uint32_t is_drain_tilizer_core_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 8: is_drain_tilizer_core

    // Add work split runtime args for tilizer cores
    uint32_t tilizer_subtoken_offset_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 9: tilizer_subtoken_offset
    uint32_t tilizer_subtoken_size_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 10: tilizer_subtoken_size

    // Token range for parallel metadata processing across tilizer cores
    uint32_t core_token_start_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 11: core_token_start
    uint32_t core_token_end_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 12: core_token_end
    uint32_t tilizer_core_idx_idx = selective_tilize_runtime_args.size();
    selective_tilize_runtime_args.push_back(0);  // 13: tilizer_core_idx (0 = drain, 1-3 = non-drain)

    // NOC coordinates for all tilizer cores (for cross-core communication)
    // Runtime args starting at index 14: [core0_noc_x, core0_noc_y, core1_noc_x, core1_noc_y, ...]
    for (uint32_t i = 0; i < t_num_cores; i++) {
        selective_tilize_runtime_args.push_back((uint32_t)tilizer_cores_physical.at(i).x);
        selective_tilize_runtime_args.push_back((uint32_t)tilizer_cores_physical.at(i).y);
    }

    // Calculate tokens per tilizer core for parallel metadata processing
    uint32_t tokens_per_tilizer_core = tokens / t_num_cores;

    // Compute kernel runtime args (separate from reader/writer)
    std::vector<uint32_t> compute_tilizer_runtime_args = {0};  // [0]: max_tiles_per_chunk (set per-core below)

    uint32_t tilizer_subtoken_offset = 0;
    for (uint32_t i = 0; i < t_num_cores; i++) {
        // First tilizer core is the drain tilizer core (has indices/scores sharded to it)
        selective_tilize_runtime_args.at(is_drain_tilizer_core_idx) = (i == 0) ? 1 : 0;

        // Set token range for this core's metadata processing
        // Each core processes a contiguous range of tokens
        uint32_t core_token_start = i * tokens_per_tilizer_core;
        uint32_t core_token_end = (i == t_num_cores - 1) ? tokens : (i + 1) * tokens_per_tilizer_core;
        selective_tilize_runtime_args.at(core_token_start_idx) = core_token_start;
        selective_tilize_runtime_args.at(core_token_end_idx) = core_token_end;
        selective_tilize_runtime_args.at(tilizer_core_idx_idx) = i;

        // Set work split parameters based on which group the core is in
        uint32_t tilizer_subtoken_size = 0;
        if (tilizer_cores_group_1.contains(t_cores.at(i))) {
            selective_tilize_runtime_args.at(tilizer_subtoken_offset_idx) = tilizer_subtoken_offset;
            tilizer_subtoken_size = tilizer_units_per_core_g1 * tilizer_subtoken_bytes_aligned;

            // Clamp to not exceed the total token size
            if (tilizer_subtoken_offset + tilizer_subtoken_size > aligned_input_page_size) {
                tilizer_subtoken_size = aligned_input_page_size - tilizer_subtoken_offset;
            }

            selective_tilize_runtime_args.at(tilizer_subtoken_size_idx) = tilizer_subtoken_size;
            tilizer_subtoken_offset += tilizer_subtoken_size;
        } else if (tilizer_cores_group_2.contains(t_cores.at(i))) {
            selective_tilize_runtime_args.at(tilizer_subtoken_offset_idx) = tilizer_subtoken_offset;
            tilizer_subtoken_size = tilizer_units_per_core_g2 * tilizer_subtoken_bytes_aligned;

            // Clamp to not exceed the total token size
            if (tilizer_subtoken_offset + tilizer_subtoken_size > aligned_input_page_size) {
                tilizer_subtoken_size = aligned_input_page_size - tilizer_subtoken_offset;
            }

            selective_tilize_runtime_args.at(tilizer_subtoken_size_idx) = tilizer_subtoken_size;
            tilizer_subtoken_offset += tilizer_subtoken_size;
        }

        // Set compute kernel runtime args - max_tiles_per_chunk based on tilizer_subtoken_size
        compute_tilizer_runtime_args.at(0) = tilizer_subtoken_size / tile_width_bytes;

        tt::tt_metal::SetRuntimeArgs(program, selective_tilize_kernel_id, t_cores.at(i), selective_tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_tilizer_kernel_id, t_cores.at(i), selective_tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_tilizer_kernel_id, t_cores.at(i), compute_tilizer_runtime_args);
    }

    return {
        std::move(program),
        {
            .selective_tilize_kernel_id = selective_tilize_kernel_id,
            .writer_tilizer_kernel_id = writer_tilizer_kernel_id,
            .compute_tilizer_kernel_id = compute_tilizer_kernel_id,
            .selective_tilize_cores = t_cores,
        }};
}

void AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllDispatchSelectiveTilizeSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t&,
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
            selective_tilize_runtime_args.at(4) = tensor_return_value[0].buffer()->address();
            selective_tilize_runtime_args.at(5) = tensor_return_value[1].buffer()->address();
            selective_tilize_runtime_args.at(6) =
                tensor_return_value[1].buffer()->address();  // TODO: (GR) placeholder for matmul_chunk_input_tensor
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl

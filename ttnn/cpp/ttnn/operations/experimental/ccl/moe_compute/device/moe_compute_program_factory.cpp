// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_compute_program_factory.hpp"
#include "moe_compute_device_operation_types.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/program.hpp>

// #include "impl/buffers/circular_buffer.hpp"
// #include "impl/program/program_impl.hpp"

namespace {

uint32_t get_num_pages_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size_st(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows_st(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

std::tuple<
    std::vector<CoreCoord>,  // T cores
    std::vector<CoreCoord>,  // MM cores
    CoreRangeSet,            // T CoreRangeSet
    CoreRangeSet,            // MM CoreRangeSet
    CoreRangeSet,            // T + MM CoreRangeSet
    CoreRange,               // T bounding box
    CoreRange,               // MM bounding box one
    CoreRange>               // MM bounding box two
get_cores(ttnn::MeshDevice* mesh_device) {
    // Cores
    const std::vector<CoreCoord> tilize_cores = {CoreCoord(5, 9), CoreCoord(5, 8), CoreCoord(5, 7), CoreCoord(5, 6)};
    const std::vector<CoreCoord> matmul_cores =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);

    std::pair<std::vector<CoreCoord>, std::vector<CoreCoord>> matmul_core_columns;
    for (const CoreCoord& matmul_core : matmul_cores) {
        if (matmul_core.x == 0) {
            matmul_core_columns.first.push_back(matmul_core);
        } else if (matmul_core.x == 4) {
            matmul_core_columns.second.push_back(matmul_core);
        } else {
            TT_FATAL(false, "Matmul cores must have x value 0 or 4, but has {}", matmul_core.x);
        }
    }

    // CoreRangeSets
    const CoreRangeSet tilize_core_range_set = CoreRangeSet(tilize_cores);
    const CoreRangeSet matmul_core_range_set = CoreRangeSet(matmul_cores);
    const CoreRangeSet tilize_matmul_core_range_set = tilize_core_range_set.merge(matmul_core_range_set);

    // Bounding boxes
    const CoreRange tilize_bounding_box = tilize_core_range_set.bounding_box();
    const CoreRange matmul_bounding_box_one = CoreRangeSet(matmul_core_columns.first).bounding_box();
    const CoreRange matmul_bounding_box_two = CoreRangeSet(matmul_core_columns.second).bounding_box();

    // Verify none of the bounding boxes overlap
    TT_FATAL(
        !tilize_bounding_box.intersects(matmul_bounding_box_one), "tilize and matmul bounding boxes cannot overlap");
    TT_FATAL(
        !tilize_bounding_box.intersects(matmul_bounding_box_two), "tilize and matmul bounding boxes cannot overlap");
    TT_FATAL(
        !matmul_bounding_box_one.intersects(matmul_bounding_box_two), "separate matmul bounding boxes cannot overlap");

    // Combine cores (16 total), that don't overlap with any of the tilize or matmul bounding boxes
    const CoreRange combine_core_range({1, 0}, {2, 7});
    const CoreRangeSet combine_core_range_set = CoreRangeSet(combine_core_range);
    const CoreRange combine_bounding_box = combine_core_range_set.bounding_box();
    TT_FATAL(!combine_bounding_box.intersects(tilize_bounding_box), "combine and tilize bounding boxes cannot overlap");
    TT_FATAL(
        !combine_bounding_box.intersects(matmul_bounding_box_one), "combine and matmul bounding boxes cannot overlap");
    TT_FATAL(
        !combine_bounding_box.intersects(matmul_bounding_box_two), "combine and matmul bounding boxes cannot overlap");

    return {
        tilize_cores,
        matmul_cores,
        tilize_core_range_set,
        matmul_core_range_set,
        tilize_matmul_core_range_set,
        tilize_bounding_box,
        matmul_bounding_box_one,
        matmul_bounding_box_two};
}

}  // namespace

namespace ttnn::experimental::prim {

MoEComputeMeshWorkloadFactory::cached_mesh_workload_t MoEComputeMeshWorkloadFactory::create_mesh_workload(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, MoEComputeMeshWorkloadFactory::shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            MoEComputeMeshWorkloadFactory::create_at(args, coord, tensor_args, tensor_return_value, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return MoEComputeMeshWorkloadFactory::cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MoEComputeMeshWorkloadFactory::shared_variables_t>
MoEComputeMeshWorkloadFactory::create_at(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet&) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Alignment
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const auto dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // Input tensors
    const ttnn::Tensor& tilize_input_tensor = tensor_args.tilize_input_tensor;
    const ttnn::Tensor& tilize_indices_tensor = tensor_args.tilize_expert_indices_tensor;
    const ttnn::Tensor& tilize_input_scores_tensor = tensor_args.tilize_expert_scores_tensor;
    const ttnn::Tensor& tilize_mapping_tensor = tensor_args.tilize_expert_mapping_tensor;

    const auto& tilize_input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    const auto& tilize_indices_shape = tilize_indices_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& tilize_input_scores_shape = tilize_input_scores_tensor.tensor_spec().logical_shape();
    const auto& tilize_mapping_shape = tilize_mapping_tensor.tensor_spec().logical_shape();

    const uint32_t tilize_input_pages = get_num_pages_st(tilize_input_tensor);
    const uint32_t tilize_indices_pages = get_num_pages_st(tilize_indices_tensor);
    const uint32_t tilize_input_scores_pages = get_num_pages_st(tilize_input_scores_tensor);
    const uint32_t tilize_mapping_pages = get_num_pages_st(tilize_mapping_tensor);

    const uint32_t tilize_input_page_size = get_page_size_st(tilize_input_tensor);
    const uint32_t tilize_indices_page_size = get_page_size_st(tilize_indices_tensor);
    [[maybe_unused]] const uint32_t tilize_input_scores_page_size = get_page_size_st(tilize_input_scores_tensor);
    const uint32_t tilize_mapping_page_size = get_page_size_st(tilize_mapping_tensor);

    const uint32_t tilize_input_aligned_page_size = get_aligned_page_size_st(tilize_input_tensor);
    const uint32_t tilize_indices_aligned_page_size = get_aligned_page_size_st(tilize_indices_tensor);
    const uint32_t tilize_input_scores_aligned_page_size = get_aligned_page_size_st(tilize_input_scores_tensor);
    const uint32_t tilize_mapping_aligned_page_size = get_aligned_page_size_st(tilize_mapping_tensor);

    [[maybe_unused]] const ttnn::Tensor& matmul_w0_w1_tensor = tensor_args.matmul_w0_w1_tensor;
    [[maybe_unused]] const ttnn::Tensor& matmul_w2_tensor = tensor_args.matmul_w2_tensor;

    // Output tensors
    const ttnn::Tensor& tilize_per_expert_total_tokens_output_tensor = tensor_return_value.at(0);
    const ttnn::Tensor& tilize_expert_activation_output_tensor = tensor_return_value.at(1);
    const ttnn::Tensor& tilize_e_t_output_tensor = tensor_return_value.at(2);
    const ttnn::Tensor& output_tensor = tensor_return_value.at(3);

    [[maybe_unused]] const auto& tilize_per_expert_total_tokens_output_shape =
        tilize_per_expert_total_tokens_output_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& tilize_expert_activation_output_shape =
        tilize_expert_activation_output_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& tilize_e_t_output_shape = tilize_e_t_output_tensor.tensor_spec().logical_shape();
    [[maybe_unused]] const auto& output_shape = output_tensor.tensor_spec().logical_shape();

    [[maybe_unused]] const uint32_t tilize_per_expert_total_tokens_output_pages =
        get_num_pages_st(tilize_per_expert_total_tokens_output_tensor);
    [[maybe_unused]] const uint32_t tilize_expert_activation_output_pages =
        get_num_pages_st(tilize_expert_activation_output_tensor);
    [[maybe_unused]] const uint32_t tilize_e_t_output_pages = get_num_pages_st(tilize_e_t_output_tensor);
    [[maybe_unused]] const uint32_t output_pages = get_num_pages_st(output_tensor);

    const uint32_t tilize_per_expert_total_tokens_output_page_size =
        get_page_size_st(tilize_per_expert_total_tokens_output_tensor);
    const uint32_t tilize_expert_activation_output_page_size = get_page_size_st(tilize_expert_activation_output_tensor);
    const uint32_t tilize_e_t_output_page_size = get_page_size_st(tilize_e_t_output_tensor);
    [[maybe_unused]] const uint32_t output_page_size = get_page_size_st(output_tensor);

    [[maybe_unused]] const uint32_t tilize_per_expert_total_tokens_output_aligned_page_size =
        get_aligned_page_size_st(tilize_per_expert_total_tokens_output_tensor);
    [[maybe_unused]] const uint32_t tilize_expert_activation_output_aligned_page_size =
        get_aligned_page_size_st(tilize_expert_activation_output_tensor);
    [[maybe_unused]] const uint32_t tilize_e_t_output_aligned_page_size =
        get_aligned_page_size_st(tilize_e_t_output_tensor);
    [[maybe_unused]] const uint32_t output_aligned_page_size = get_aligned_page_size_st(output_tensor);

    // Mesh
    auto* mesh_device = tilize_input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    uint32_t num_devices = mesh_view.num_devices();
    uint32_t linearized_mesh_coord = ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    // Tilize output, matmul input
    const auto tilize_output_dtype = tilize_input_tensor.dtype();
    const auto tilize_output_dataformat = tt::tt_metal::datatype_to_dataformat_converter(tilize_output_dtype);
    const uint32_t tilize_output_page_size = tt::tile_size(tilize_output_dataformat);

    // General info
    uint32_t tokens = get_num_rows_st(tilize_input_tensor);
    uint32_t hidden_size = tilize_input_shape[-1];
    uint32_t experts = tilize_mapping_shape[-1];
    uint32_t selected_experts_k = tilize_indices_shape[-1];
    uint32_t experts_per_device = tt::div_up(experts, num_devices);

    // Cores
    const auto
        [tilize_cores,
         matmul_cores,
         tilize_core_range_set,
         matmul_core_range_set,
         tilize_matmul_core_range_set,
         tilize_bounding_box,
         matmul_bounding_box_one,
         matmul_bounding_box_two] = get_cores(mesh_device);

    const uint32_t tilize_num_cores = tilize_core_range_set.num_cores();
    const uint32_t matmul_num_cores = matmul_core_range_set.num_cores();

    const uint32_t matmul_bounding_box_one_num_cores = matmul_bounding_box_one.size();
    const uint32_t matmul_bounding_box_two_num_cores = matmul_bounding_box_two.size();

    // Logical mcast bounding box coordinates
    const CoreCoord tilize_mcast_start_logical = tilize_bounding_box.start_coord;
    const CoreCoord tilize_mcast_end_logical = tilize_bounding_box.end_coord;
    const CoreCoord matmul_mcast_box_one_start_logical = matmul_bounding_box_one.start_coord;
    const CoreCoord matmul_mcast_box_one_end_logical = matmul_bounding_box_one.end_coord;
    const CoreCoord matmul_mcast_box_two_start_logical = matmul_bounding_box_two.start_coord;
    const CoreCoord matmul_mcast_box_two_end_logical = matmul_bounding_box_two.end_coord;

    // Convert to physical NOC coordinates
    const CoreCoord tilize_mcast_start_physical =
        mesh_device->worker_core_from_logical_core(tilize_mcast_start_logical);
    const CoreCoord tilize_mcast_end_physical = mesh_device->worker_core_from_logical_core(tilize_mcast_end_logical);
    const CoreCoord matmul_mcast_box_one_start_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_box_one_start_logical);
    const CoreCoord matmul_mcast_box_one_end_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_box_one_end_logical);
    const CoreCoord matmul_mcast_box_two_start_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_box_two_start_logical);
    const CoreCoord matmul_mcast_box_two_end_physical =
        mesh_device->worker_core_from_logical_core(matmul_mcast_box_two_end_logical);

    //-------------------------------------------------------------------------
    // Tilize semaphores
    //-------------------------------------------------------------------------

    // Non-drain-sync cores signal to drain-sync core that partial metadata results are ready
    auto tilize_partial_metadata_ready_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    // Non-drain-sync cores signal to drain-sync core that partial chunk has been sent to the matmul cores.
    // Drain-sync core then signals to non-drain-sync core that they can begin sending the next chunk.
    auto tilize_chunk_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    // Since both reader and writer are using NoC1, we have the readers wait until all writers have sent their
    // chunk portion to the matmul cores before reading in another set of tokens to tilize. This frees up NoC1
    // to do the mcast, which is also a requirement for doing linked mcasts.
    auto previous_chunk_sent_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Matmul semaphores
    //-------------------------------------------------------------------------

    // Create semaphores for ring synchronization between cores
    // Each core will have a semaphore that its predecessor will signal
    const uint32_t ring_semaphore_id = tt::tt_metal::CreateSemaphore(program, matmul_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Tilize and Matmul semaphores
    //-------------------------------------------------------------------------

    // Tilize drain-sync core signals to tilize non-drain-sync cores that final metadata results are ready
    // Tilize drain-sync core signals to matmul cores that final metadata results are ready
    auto metadata_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    // Matmul cores signal to tilize drain-sync-core that the input chunk is free to be written to
    // Tilize drain-sync-core propogates this to the tilize non-drain-sync cores
    auto matmul_chunk_available_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    // Tilize drain-sync core signals to all matmul cores that a full chunk is ready
    auto matmul_chunk_ready_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, tilize_matmul_core_range_set, INVALID);

    //-------------------------------------------------------------------------
    // Tilize work split
    //-------------------------------------------------------------------------

    // Split token subregions across tilize cores (similar to sender subtoken splitting)
    // Each tilize core handles a portion of the hidden dimension for each token
    uint32_t tilize_subtoken_bytes_aligned =
        tt::align(tt::div_up(tilize_input_aligned_page_size, tilize_num_cores), l1_alignment);
    uint32_t tilize_subtoken_units_of_work = tt::div_up(tilize_input_aligned_page_size, tilize_subtoken_bytes_aligned);

    auto
        [num_tilize_work_cores,
         all_tilize_work_cores,
         tilize_cores_group_1,
         tilize_cores_group_2,
         tilize_units_per_core_g1,
         tilize_units_per_core_g2] =
            tt::tt_metal::split_work_to_cores(tilize_core_range_set, tilize_subtoken_units_of_work);

    //-------------------------------------------------------------------------
    // Tilize and Matmul CBs
    //-------------------------------------------------------------------------

    /*
     * Tilize: Used as output CB of tilize operation
     * MM: Used as input CB (where tilized chunks arrive)
     * Combine: Stores output of MM, for input to combine
     */
    uint32_t tilize_output_cb_id = tt::CBIndex::c_0;
    [[maybe_unused]] uint32_t cb_s2c_in_id = tt::CBIndex::c_0;
    [[maybe_unused]] uint32_t combine_input_cb_id = tt::CBIndex::c_0;

    // All cores (not just Tilize and Matmul)
    CoreRangeSet shard_cores = output_tensor.memory_config().shard_spec()->grid;
    auto output_cb = tt::tt_metal::create_cb(
        tilize_output_cb_id,
        program,
        shard_cores,
        output_page_size,
        output_pages / shard_cores.size(),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()),
        output_tensor.buffer());
    tt::tt_metal::CBHandle sharded_output_cb_handle = std::get<1>(output_cb);

    //-------------------------------------------------------------------------
    // Tilize CBs
    //-------------------------------------------------------------------------

    // after determining the total number of tokens for each expert, this buffer will store the total number of tokens
    // for each expert to pass to the other kernels
    uint32_t per_expert_total_tokens_cb_id = tt::CBIndex::c_1;
    // CB for passing total_chunks from writer to compute
    uint32_t total_chunks_cb_id = tt::CBIndex::c_2;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_3;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_4;
    // full scores buffer
    uint32_t scores_tensor_cb_id = tt::CBIndex::c_5;
    // Send preparation buffer [E, T] for untilize, capped by -1 to indicate no more tokens to send for this expert
    uint32_t e_t_cb_id = tt::CBIndex::c_6;
    // tilize input buffer for tokens to be tilized (row-major from reader)
    uint32_t tilize_input_cb_id = tt::CBIndex::c_7;
    // Experts activation buffer [T, 2*E + 1] each row is {token id, expert_0_activated, expert_1_activated,...,
    // expert_0_score, expert_1_score, ...} k+1 if not activated, k value in the indices tensor for that token if
    // activated
    uint32_t expert_activation_cb_id = tt::CBIndex::c_8;
    // BRISC's e_t buffer for parallel metadata processing (BRISC processes tokens/2 to tokens)
    uint32_t brisc_e_t_cb_id = tt::CBIndex::c_19;
    // BRISC's per-expert token counts to communicate to NCRISC after parallel processing
    uint32_t brisc_expert_counts_cb_id = tt::CBIndex::c_10;
    // BRISC's expert activation buffer for parallel processing
    uint32_t brisc_expert_activation_cb_id = tt::CBIndex::c_11;
    // BRISC's activated token count (single uint32_t)
    uint32_t brisc_activated_count_cb_id = tt::CBIndex::c_12;

    uint32_t remote_counts_cb_id = tt::CBIndex::c_13;

    const auto tilize_input_data_format = tt::tt_metal::datatype_to_dataformat_converter(tilize_input_tensor.dtype());
    const auto tilize_indices_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_indices_tensor.dtype());
    const auto tilize_input_scores_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_input_scores_tensor.dtype());
    const auto tilize_mapping_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_mapping_tensor.dtype());
    const auto tilize_e_t_output_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tilize_e_t_output_tensor.dtype());

    uint32_t max_tilize_subtoken_size =
        std::max(tilize_units_per_core_g1, tilize_units_per_core_g2) * tilize_subtoken_bytes_aligned;

    constexpr uint32_t tokens_per_chunk = 32;  // Hardcoding for now, can adjust when tiny tiles support added

    tt::tt_metal::create_cb(
        per_expert_total_tokens_cb_id,
        program,
        tilize_core_range_set,
        tilize_per_expert_total_tokens_output_page_size,
        1,
        tt::tt_metal::datatype_to_dataformat_converter(tilize_per_expert_total_tokens_output_tensor.dtype()));

    // e_t buffer entry size must be 16B aligned for NOC DMA during BRISC->NCRISC merge
    tt::tt_metal::create_cb(
        e_t_cb_id,
        program,
        tilize_core_range_set,
        tilize_e_t_output_page_size,
        experts_per_device,  // number of experts on the device
        tilize_e_t_output_data_format);

    // Assume indices tensor is sharded in L1
    tt::tt_metal::create_cb(
        indices_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_indices_aligned_page_size,
        tilize_indices_pages,  // double buffer buffer packets
        tilize_indices_data_format,
        tilize_indices_tensor.buffer());

    // Assume scores tensor is sharded in L1
    tt::tt_metal::create_cb(
        scores_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_input_scores_aligned_page_size,
        tilize_input_scores_pages,
        tilize_input_scores_data_format,
        tilize_input_scores_tensor.buffer());

    // For each batch's tokens, we need to read the relevant experts from the mapping tensor
    // For in range (tokens) every time tokens/batch increments, read in new mapping tensor page
    tt::tt_metal::create_cb(
        mapping_tensor_cb_id,
        program,
        tilize_core_range_set,
        tilize_mapping_aligned_page_size,
        tilize_mapping_pages,
        tilize_mapping_data_format);

    // Tilize input buffer: holds subtokens for tokens_per_chunk tokens
    // Each tilize core reads its subtoken portion of incoming tokens
    // Single buffered so we don't start reading in next set of tokens
    // before previous chunk fully sent to matmul cores
    tt::tt_metal::create_cb(
        tilize_input_cb_id,
        program,
        tilize_core_range_set,
        max_tilize_subtoken_size,
        tokens_per_chunk,
        tilize_input_data_format);

    tt::tt_metal::create_cb(
        expert_activation_cb_id,
        program,
        tilize_core_range_set,
        tt::align((2 * experts_per_device + 1) * sizeof(uint32_t), l1_alignment),
        tokens,
        tt::DataFormat::UInt32);

    // BRISC's e_t buffer for parallel metadata processing
    // BRISC processes the second half of tokens (tokens/2 to tokens)
    // Single page containing all experts' token lists, each with capacity tokens/2
    // Uses same 16B entry alignment as main e_t buffer for NOC DMA compatibility
    tt::tt_metal::create_cb(
        brisc_e_t_cb_id,
        program,
        tilize_core_range_set,
        (tokens / 2) * l1_alignment * experts_per_device,  // full buffer with 16B entries
        1,
        tt::DataFormat::UInt32);

    // BRISC's per-expert token counts to communicate to NCRISC
    // Single page containing counts for all experts
    tt::tt_metal::create_cb(
        brisc_expert_counts_cb_id,
        program,
        tilize_core_range_set,
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
        tilize_core_range_set,
        brisc_activation_row_size * (tokens / 2),  // full buffer in one page
        1,
        tt::DataFormat::UInt32);

    // BRISC's activated token count (single uint32_t to communicate to NCRISC)
    tt::tt_metal::create_cb(
        brisc_activated_count_cb_id, program, tilize_core_range_set, sizeof(uint32_t), 1, tt::DataFormat::UInt32);

    // CB for receiving counts from non-drain tilize cores (only used on drain core)
    // Each non-drain core sends: [e_t_count_expert0, e_t_count_expert1, activated_count]
    // Layout: 3 values per core × (tilize_num_cores - 1) cores, 16B aligned per core's data
    uint32_t counts_per_remote_core = experts_per_device + 1;  // e_t counts + activated count
    uint32_t remote_counts_entry_size = tt::align(counts_per_remote_core * sizeof(uint32_t), l1_alignment);
    tt::tt_metal::create_cb(
        remote_counts_cb_id,
        program,
        tilize_core_range_set,
        remote_counts_entry_size,
        tilize_num_cores - 1,  // one entry per non-drain core
        tt::DataFormat::UInt32);

    // CB for passing total_chunks from writer to compute kernel
    // Single page holding one uint32_t value
    tt::tt_metal::create_cb(
        total_chunks_cb_id,
        program,
        tilize_core_range_set,
        sizeof(uint32_t),
        1,  // single page
        tt::DataFormat::UInt32);

    //-------------------------------------------------------------------------
    // Matmul CBs
    //-------------------------------------------------------------------------

    // CBs used in the MOE operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_s2c_in      | CBIndex::c_0  | Float16_b  | true  |    224*2 |      917504     |
        | cb_r2c_w0      | CBIndex::c_1  | Bfp4_b     | true  |    14*6  |      48384      |
        | cb_c2w_rdy     | CBIndex::c_2  | Float32    | false |    1     |      4          |
        | cb_w2c_rdy     | CBIndex::c_3  | Float32    | false |    1     |      4          |
        | cb_s2c_in2     | CBIndex::c_4  | Float16_b  | true  |    6*2   |      24576      |
        | cb_w2c_md      | CBIndex::c_5  | UInt32     | false |    2     |      8          |
        ------------------------------------------------------------------------------------
    */

    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb
    // Note: cb_s2c_in is handled separately as it is allocated on Tilize, Matmul, and (future) Combine cores
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t>> matmul_cb_specs0 = {
        {"cb_r2c_w0", tt::CBIndex::c_1, tt::DataFormat::Bfp4_b, true, 14 * 6},
        {"cb_c2w_rdy", tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1},
        {"cb_w2c_rdy", tt::CBIndex::c_3, tt::DataFormat::Float32, false, 1},
        {"cb_s2c_in2", tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 6 * 2},
        {"cb_w2c_md", tt::CBIndex::c_5, tt::DataFormat::UInt32, false, 2},
    };

    std::map<std::string, tt::tt_metal::CBHandle> matmul_cb_handles;

    // Create CBs
    for (const auto& [name, index, data_format, is_tile, tiles_per_cb] : matmul_cb_specs0) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        const auto cb_config = tt::tt_metal::CircularBufferConfig(tiles_per_cb * bytes_per_tile, {{index, data_format}})
                                   .set_page_size(index, bytes_per_tile);

        matmul_cb_handles[name] = tt::tt_metal::CreateCircularBuffer(program, matmul_core_range_set, cb_config);
    }

    //-------------------------------------------------------------------------
    // Tilize kernels
    //-------------------------------------------------------------------------

    // For NOC 0: start = (min_x, min_y), end = (max_x, max_y)
    // For NOC 1: coordinates are swapped
    // We'll use NOC 0 by default, but pass both orderings and let the kernel handle it
    // Or we can determine the NOC here and swap if needed
    // For simplicity, we pass the NOC 0 ordering (start < end) and the kernel will use NOC 0

    // Store physical NOC coordinates of all tilize cores for cross-core communication
    // Used by drain core to read from non-drain cores, and non-drain to write to drain
    std::vector<CoreCoord> tilize_cores_physical;
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        tilize_cores_physical.push_back(mesh_device->worker_core_from_logical_core(tilize_cores.at(i)));
    }

    // tile_width_bytes = TILE_WIDTH * element_size
    // max_tiles_per_chunk = max_tilize_subtoken_size / tile_width_bytes
    constexpr uint32_t TILE_WIDTH = 32;
    uint32_t tile_width_bytes = TILE_WIDTH * tilize_input_tensor.element_size();
    uint32_t max_tiles_per_chunk = max_tilize_subtoken_size / tile_width_bytes;

    // Drain core is always the first tilize core (index 0)
    CoreCoord tilize_drain_core_physical = tilize_cores_physical.at(0);

    std::unordered_map<std::string, uint32_t> tilize_named_compile_time_args = {
        // CBs
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"tilize_output_cb_id", tilize_output_cb_id},
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
        {"e_t_entry_size", l1_alignment},

        // Number of pages
        {"input_pages", tilize_input_pages},
        {"indices_pages", tilize_indices_pages},
        {"mapping_pages", tilize_mapping_pages},
        {"scores_pages", tilize_input_scores_pages},

        // Page sizes
        {"input_page_size", tilize_input_page_size},
        {"indices_page_size", tilize_indices_page_size},
        {"mapping_page_size", tilize_mapping_page_size},
        {"per_expert_total_tokens_output_page_size", tilize_per_expert_total_tokens_output_page_size},
        {"expert_activation_output_page_size", tilize_expert_activation_output_page_size},
        {"e_t_output_page_size", tilize_e_t_output_page_size},
        {"tilize_output_page_size", tilize_output_page_size},

        // Aligned page sizes
        {"aligned_input_page_size", tilize_input_aligned_page_size},
        {"aligned_indices_page_size", tilize_indices_aligned_page_size},
        {"aligned_mapping_page_size", tilize_mapping_aligned_page_size},
        {"aligned_scores_page_size", tilize_input_scores_aligned_page_size},

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
        {"cluster_axis", (uint32_t)args.cluster_axis.value()},

        // Coordinates for non-drain-sync to drain-sync synchronization
        {"drain_core_noc_x", (uint32_t)tilize_drain_core_physical.x},
        {"drain_core_noc_y", (uint32_t)tilize_drain_core_physical.y},

        // T multicast coordinates
        {"tilize_mcast_start_x", (uint32_t)tilize_mcast_start_physical.x},
        {"tilize_mcast_start_y", (uint32_t)tilize_mcast_start_physical.y},
        {"tilize_mcast_end_x", (uint32_t)tilize_mcast_end_physical.x},
        {"tilize_mcast_end_y", (uint32_t)tilize_mcast_end_physical.y},
        {"num_tilize_cores", tilize_num_cores},

        // MM multicast coordinates
        {"num_matmul_cores", matmul_num_cores},

        {"matmul_mcast_box_one_start_x", (uint32_t)matmul_mcast_box_one_start_physical.x},
        {"matmul_mcast_box_one_start_y", (uint32_t)matmul_mcast_box_one_start_physical.y},
        {"matmul_mcast_box_one_end_x", (uint32_t)matmul_mcast_box_one_end_physical.x},
        {"matmul_mcast_box_one_end_y", (uint32_t)matmul_mcast_box_one_end_physical.y},
        {"num_matmul_bounding_box_one_cores", matmul_bounding_box_one_num_cores},

        {"matmul_mcast_box_two_start_x", (uint32_t)matmul_mcast_box_two_start_physical.x},
        {"matmul_mcast_box_two_start_y", (uint32_t)matmul_mcast_box_two_start_physical.y},
        {"matmul_mcast_box_two_end_x", (uint32_t)matmul_mcast_box_two_end_physical.x},
        {"matmul_mcast_box_two_end_y", (uint32_t)matmul_mcast_box_two_end_physical.y},
        {"num_matmul_bounding_box_two_cores", matmul_bounding_box_two_num_cores},

        // Semaphores
        {"partial_metadata_ready_semaphore_id", tilize_partial_metadata_ready_semaphore_id},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
        {"tilize_chunk_ready_semaphore_id", tilize_chunk_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
        {"previous_chunk_sent_semaphore_id", previous_chunk_sent_semaphore_id},
    };

    std::vector<uint32_t> tilize_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(tilize_input_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_indices_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_input_scores_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_mapping_tensor.buffer()).append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_per_expert_total_tokens_output_tensor.buffer())
        .append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_expert_activation_output_tensor.buffer())
        .append_to(tilize_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tilize_e_t_output_tensor.buffer()).append_to(tilize_compile_time_args);

    tt::tt_metal::KernelHandle tilize_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp",
        tilize_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = tilize_compile_time_args,
            .defines = {},
            .named_compile_args = tilize_named_compile_time_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

    tt::tt_metal::KernelHandle tilize_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_writer.cpp",
        tilize_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = tilize_compile_time_args,
            .defines = {},
            .named_compile_args = tilize_named_compile_time_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

    // Compute kernel compile-time args for tilization
    std::unordered_map<std::string, uint32_t> compute_tilize_named_compile_time_args = {
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"total_chunks_cb_id", total_chunks_cb_id},
        {"tokens_per_chunk", tokens_per_chunk},
        {"max_tiles_per_chunk", max_tiles_per_chunk},
    };

    tt::tt_metal::KernelHandle tilize_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_compute.cpp",
        tilize_core_range_set,
        tt::tt_metal::ComputeConfig{.named_compile_args = compute_tilize_named_compile_time_args});

    std::vector<uint32_t> tilize_runtime_args = {
        tilize_input_tensor.buffer()->address(),                           // 0
        tilize_indices_tensor.buffer()->address(),                         // 1
        tilize_input_scores_tensor.buffer()->address(),                    // 2
        tilize_mapping_tensor.buffer()->address(),                         // 3
        tilize_per_expert_total_tokens_output_tensor.buffer()->address(),  // 4
        tilize_expert_activation_output_tensor.buffer()->address(),        // 5
        tilize_e_t_output_tensor.buffer()->address(),                      // 6
    };

    uint32_t is_drain_tilize_core_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 8: is_drain_tilize_core

    // Add work split runtime args for tilize cores
    uint32_t tilize_subtoken_offset_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 9: tilize_subtoken_offset
    uint32_t tilize_subtoken_size_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 10: tilize_subtoken_size

    // Token range for parallel metadata processing across tilize cores
    uint32_t core_token_start_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 11: core_token_start
    uint32_t core_token_end_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 12: core_token_end
    uint32_t tilize_core_idx_idx = tilize_runtime_args.size();
    tilize_runtime_args.push_back(0);  // 13: tilize_core_idx (0 = drain, 1-3 = non-drain)

    // NOC coordinates for all tilize cores (for cross-core communication)
    // Runtime args starting at index 14: [core0_noc_x, core0_noc_y, core1_noc_x, core1_noc_y, ...]
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(i).x);
        tilize_runtime_args.push_back((uint32_t)tilize_cores_physical.at(i).y);
    }

    // Calculate tokens per tilize core for parallel metadata processing
    uint32_t tokens_per_tilize_core = tokens / tilize_num_cores;

    // Compute kernel runtime args (separate from reader/writer)
    std::vector<uint32_t> tilize_compute_runtime_args = {0};  // [0]: max_tiles_per_chunk (set per-core below)

    uint32_t tilize_subtoken_offset = 0;
    for (uint32_t i = 0; i < tilize_num_cores; i++) {
        // First tilize core is the drain tilize core (has indices/scores sharded to it)
        tilize_runtime_args.at(is_drain_tilize_core_idx) = (i == 0) ? 1 : 0;

        // Set token range for this core's metadata processing
        // Each core processes a contiguous range of tokens
        uint32_t core_token_start = i * tokens_per_tilize_core;
        uint32_t core_token_end = (i == tilize_num_cores - 1) ? tokens : (i + 1) * tokens_per_tilize_core;
        tilize_runtime_args.at(core_token_start_idx) = core_token_start;
        tilize_runtime_args.at(core_token_end_idx) = core_token_end;
        tilize_runtime_args.at(tilize_core_idx_idx) = i;

        // Set work split parameters based on which group the core is in
        uint32_t tilize_subtoken_size = 0;
        if (tilize_cores_group_1.contains(tilize_cores.at(i))) {
            tilize_runtime_args.at(tilize_subtoken_offset_idx) = tilize_subtoken_offset;
            tilize_subtoken_size = tilize_units_per_core_g1 * tilize_subtoken_bytes_aligned;

            // Clamp to not exceed the total token size
            if (tilize_subtoken_offset + tilize_subtoken_size > tilize_input_aligned_page_size) {
                tilize_subtoken_size = tilize_input_aligned_page_size - tilize_subtoken_offset;
            }

            tilize_runtime_args.at(tilize_subtoken_size_idx) = tilize_subtoken_size;
            tilize_subtoken_offset += tilize_subtoken_size;
        } else if (tilize_cores_group_2.contains(tilize_cores.at(i))) {
            tilize_runtime_args.at(tilize_subtoken_offset_idx) = tilize_subtoken_offset;
            tilize_subtoken_size = tilize_units_per_core_g2 * tilize_subtoken_bytes_aligned;

            // Clamp to not exceed the total token size
            if (tilize_subtoken_offset + tilize_subtoken_size > tilize_input_aligned_page_size) {
                tilize_subtoken_size = tilize_input_aligned_page_size - tilize_subtoken_offset;
            }

            tilize_runtime_args.at(tilize_subtoken_size_idx) = tilize_subtoken_size;
            tilize_subtoken_offset += tilize_subtoken_size;
        }

        // Set compute kernel runtime args - max_tiles_per_chunk based on tilize_subtoken_size
        tilize_compute_runtime_args.at(0) = tilize_subtoken_size / tile_width_bytes;

        tt::tt_metal::SetRuntimeArgs(program, tilize_reader_kernel_id, tilize_cores.at(i), tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, tilize_writer_kernel_id, tilize_cores.at(i), tilize_runtime_args);
        tt::tt_metal::SetRuntimeArgs(
            program, tilize_compute_kernel_id, tilize_cores.at(i), tilize_compute_runtime_args);
    }

    //-------------------------------------------------------------------------
    // Matmul kernels
    //-------------------------------------------------------------------------

    // Create compile args for the program
    const auto matmul_tensors =
        std::vector<const ttnn::Tensor*>{&matmul_w0_w1_tensor, &matmul_w2_tensor, &output_tensor};

    std::vector<uint32_t> matmul_compile_time_args;
    for (const auto& tensor : matmul_tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(matmul_compile_time_args);
    }

    std::unordered_map<std::string, uint32_t> matmul_named_compile_time_args = {
        {"num_experts", experts_per_device},
        {"layer_id", args.layer_id},
        {"num_cores", static_cast<uint32_t>(matmul_num_cores)},
        {"metadata_ready_semaphore_id", metadata_ready_semaphore_id},
        {"matmul_chunk_ready_semaphore_id", matmul_chunk_ready_semaphore_id},
        {"matmul_chunk_available_semaphore_id", matmul_chunk_available_semaphore_id},
        {"per_expert_total_tokens_cb_id", per_expert_total_tokens_cb_id},
        {"tokens_per_chunk", tokens_per_chunk},
        {"tilize_drain_core_noc_x", (uint32_t)tilize_drain_core_physical.x},
        {"tilize_drain_core_noc_y", (uint32_t)tilize_drain_core_physical.y},
    };

    // Create kernels for the program
    auto matmul_dm0_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm0.cpp",
        matmul_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = matmul_compile_time_args,
            .named_compile_args = matmul_named_compile_time_args});

    auto matmul_dm1_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp",
        matmul_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = matmul_compile_time_args,
            .named_compile_args = matmul_named_compile_time_args});

    auto matmul_compute_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp",
        matmul_core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .bfp8_pack_precise = false,
            .math_approx_mode = true,
            .compile_args = matmul_compile_time_args,
            .named_compile_args = matmul_named_compile_time_args});

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    // Sort cores by (descending y, descending x) to create a ring that flows naturally
    std::vector<uint32_t> ring_pos2bank_id(matmul_num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [mesh_device, &matmul_cores](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = mesh_device->worker_core_from_logical_core(matmul_cores[bank_id_a]);
            const auto& pb = mesh_device->worker_core_from_logical_core(matmul_cores[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;  // Descending y
            }
            return pa.x > pb.x;  // Descending x
        });

    // Build a map where key = bank_id, value = {ring position (i), neighbor's bank_id}
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> bank2ring_pos;
    for (uint32_t ring_pos = 0; ring_pos < matmul_num_cores; ++ring_pos) {
        uint32_t this_bank = ring_pos2bank_id[ring_pos];
        uint32_t next_bank = ring_pos2bank_id[(ring_pos + 1) % matmul_num_cores];
        bank2ring_pos[this_bank] = {ring_pos, next_bank};
    }

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> matmul_runtime_args;
    matmul_runtime_args.push_back(0);  // DRAM Bank ID placeholder
    matmul_runtime_args.push_back(0);  // VChannel placeholder
    for (const auto& tensor : matmul_tensors) {
        matmul_runtime_args.push_back(tensor->buffer()->address());
    }
    // Add placeholders for neighbor physical coords and semaphore
    matmul_runtime_args.push_back(ring_semaphore_id);  // Semaphore ID
    matmul_runtime_args.push_back(0);                  // Ring core ID placeholder
    matmul_runtime_args.push_back(0);                  // Neighbor physical x
    matmul_runtime_args.push_back(0);                  // Neighbor physical y

    std::vector<uint32_t> vchannels;
    uint32_t dram_bank = 0;
    for (auto core : matmul_cores) {
        uint32_t vchannel = dram_bank & 0x3;

        // Check if there is any core with the same row
        auto it = std::find_if(matmul_cores.begin(), matmul_cores.begin() + dram_bank, [&](const auto& core_prev) {
            return core_prev.y == core.y;
        });

        // If there is any core with the same row, make sure the VChannel is different
        if (it != matmul_cores.begin() + dram_bank) {
            size_t j = std::distance(matmul_cores.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        // Use the optimized ring neighbor mapping
        const auto [ring_pos, next_bank] = bank2ring_pos[dram_bank];
        const auto& next_physical = mesh_device->worker_core_from_logical_core(matmul_cores[next_bank]);

        matmul_runtime_args[0] = dram_bank++;
        matmul_runtime_args[1] = vchannel;
        // matmul_runtime_args[2-4] are already set to tensor addresses
        // matmul_runtime_args[5] is already set to ring_semaphore_id
        matmul_runtime_args[6] = ring_pos;
        matmul_runtime_args[7] = static_cast<uint32_t>(next_physical.x);
        matmul_runtime_args[8] = static_cast<uint32_t>(next_physical.y);

        tt::tt_metal::SetRuntimeArgs(program, matmul_dm0_kernel_handle, core, matmul_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, matmul_dm1_kernel_handle, core, matmul_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, matmul_compute_kernel_handle, core, matmul_runtime_args);

        log_debug(tt::LogOp, "{} -> DRAM {} -> ring pos {}", core.str(), dram_bank, ring_pos);
    }

    //-------------------------------------------------------------------------
    // Cached program
    //-------------------------------------------------------------------------

    return {
        std::move(program),
        {.tilize_kernel_handles = {tilize_reader_kernel_id, tilize_compute_kernel_id, tilize_writer_kernel_id},
         .tilize_cores = tilize_cores,
         .matmul_kernel_handles = {matmul_dm0_kernel_handle, matmul_dm1_kernel_handle, matmul_compute_kernel_handle},
         .matmul_cores = matmul_cores,
         .sharded_output_cb_handle = sharded_output_cb_handle}};
}

void MoEComputeMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MoEComputeParams&,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    // output tensors
    const ttnn::Tensor& tilize_per_expert_total_tokens_output_tensor = tensor_return_value.at(0);
    const ttnn::Tensor& tilize_expert_activation_output_tensor = tensor_return_value.at(1);
    const ttnn::Tensor& tilize_e_t_output_tensor = tensor_return_value.at(2);
    const ttnn::Tensor& output_tensor = tensor_return_value.at(3);

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        // Update sharded circular buffer address
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_variables.sharded_output_cb_handle, *output_tensor.buffer());

        //-------------------------------------------------------------------------
        // Tilize
        //-------------------------------------------------------------------------
        for (const auto& core : shared_variables.tilize_cores) {
            // reader
            auto& tilize_reader_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles.at(0), core);
            tilize_reader_runtime_args.at(0) = tensor_args.tilize_input_tensor.buffer()->address();
            tilize_reader_runtime_args.at(1) = tensor_args.tilize_expert_indices_tensor.buffer()->address();
            tilize_reader_runtime_args.at(2) = tensor_args.tilize_expert_scores_tensor.buffer()->address();
            tilize_reader_runtime_args.at(3) = tensor_args.tilize_expert_mapping_tensor.buffer()->address();
            tilize_reader_runtime_args.at(4) = tilize_per_expert_total_tokens_output_tensor.buffer()->address();
            tilize_reader_runtime_args.at(5) = tilize_expert_activation_output_tensor.buffer()->address();
            tilize_reader_runtime_args.at(6) = tilize_e_t_output_tensor.buffer()->address();

            // writer
            auto& tilize_writer_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.tilize_kernel_handles.at(2), core);
            tilize_writer_runtime_args.at(0) = tensor_args.tilize_input_tensor.buffer()->address();
            tilize_writer_runtime_args.at(1) = tensor_args.tilize_expert_indices_tensor.buffer()->address();
            tilize_writer_runtime_args.at(2) = tensor_args.tilize_expert_scores_tensor.buffer()->address();
            tilize_writer_runtime_args.at(3) = tensor_args.tilize_expert_mapping_tensor.buffer()->address();
            tilize_writer_runtime_args.at(4) = tilize_per_expert_total_tokens_output_tensor.buffer()->address();
            tilize_writer_runtime_args.at(5) = tilize_expert_activation_output_tensor.buffer()->address();
            tilize_writer_runtime_args.at(6) = tilize_e_t_output_tensor.buffer()->address();
        }

        //-------------------------------------------------------------------------
        // Matmul
        //-------------------------------------------------------------------------
        for (const auto& core : shared_variables.matmul_cores) {
            for (const auto& kernel_handle : shared_variables.matmul_kernel_handles) {
                auto& matmul_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_handle, core);
                matmul_runtime_args.at(2) = tensor_args.matmul_w0_w1_tensor.buffer()->address();
                matmul_runtime_args.at(3) = tensor_args.matmul_w2_tensor.buffer()->address();
                matmul_runtime_args.at(4) = output_tensor.buffer()->address();
            }
        }
    }
}

}  // namespace ttnn::experimental::prim

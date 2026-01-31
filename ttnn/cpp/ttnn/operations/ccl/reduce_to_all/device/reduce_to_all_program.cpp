// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "reduce_to_all_op.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::ccl {

inline void fabric_mux_ct_args(
    const uint32_t num_workers_per_direction,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));  // fabric_mux_num_buffers_per_channel
    worker_ct_args.push_back(
        mux_kernel_config.get_buffer_size_bytes(channel_type));        // fabric_mux_channel_buffer_size_bytes
    worker_ct_args.push_back(mux_kernel_config.get_status_address());  // fabric_mux_status_address
    worker_ct_args.push_back(
        mux_kernel_config.get_termination_signal_address());  // fabric_mux_termination_signal_address
    worker_ct_args.push_back(num_workers_per_direction);      // num_mux_clients
}

inline void fabric_mux_rt_args(
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    uint32_t shared_termination_sync_sem,  // Shared semaphore on master's core for all mux workers
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(is_termination_master);  // is_termination_master
    worker_rt_args.push_back(mux_virtual_core.x);     // fabric_mux_x
    worker_rt_args.push_back(mux_virtual_core.y);     // fabric_mux_y
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_base_address(channel_type, worker_id));  // fabric_mux_channel_base_address
    worker_rt_args.push_back(
        mux_kernel_config.get_connection_info_address(channel_type, worker_id));  // fabric_mux_connection_info_address
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(
        channel_type, worker_id));  // fabric_mux_connection_handshake_address
    worker_rt_args.push_back(
        mux_kernel_config.get_flow_control_address(channel_type, worker_id));  // fabric_mux_flow_control_address
    worker_rt_args.push_back(
        mux_kernel_config.get_buffer_index_address(channel_type, worker_id));  // fabric_mux_buffer_index_address
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));  // fabric_mux_channel_id
    worker_rt_args.push_back(shared_termination_sync_sem);  // termination_sync_address (shared on master's core)
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);                   // termination_master_noc_x
    worker_rt_args.push_back(termination_master_virtual_core.y);                   // termination_master_noc_y
}

ttnn::device_operation::CachedProgram<ReduceToAllOp::ReduceToAll::shared_variables_t> reduce_to_all_program_factory(
    const ReduceToAllOp::tensor_args_t& tensor_args,
    const ReduceToAllOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& root_coord,
    const float scale_fp32,
    const MeshCoordinate& device_coordinate,
    std::optional<ttnn::MeshCoordinate>& forward_coord,
    std::optional<ttnn::MeshCoordinate>& backward_coord,
    ReduceToAllOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    // Reduce to all : reduce to root scheme 3 (with Ring)
    // The output of the SDPA reduction is avaialble on all devices at the end
    // Round 1:
    // devices 0-1 and devices 2-3 exchange their local data and compute the partial reduction locally
    // Round 2:
    // devices 0-3 and devices 1-2 exchange their partial reductions and compute the final result locally

    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor_l.device());
    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_ms = tensor_args.input_tensor_ms;  // Combined: col 0 = max, col 1 = sum
    const auto& fw_intermediate_tensor = output_tensors.at(0)[0];
    const auto& bw_intermediate_tensor = output_tensors.at(0)[1];
    const auto& round1_intermediate_tensor = output_tensors.at(0)[2];

    const auto& output_tensor_l = output_tensors.at(1)[0];  // Normalized L (only output)

    constexpr auto num_links = 2;
    auto* device = input_tensor_l.device();
    auto mesh_shape = mesh_device->shape();
    bool is_root2_device = false;
    bool is_sender_device = false;
    bool is_leftmost = false;
    if (device_coordinate != root_coord && backward_coord.has_value() && backward_coord.value() == root_coord) {
        // this is the intermediate device
        is_root2_device = true;
    } else if (forward_coord.has_value() && forward_coord.value() == root_coord) {
        // this is a sender device
        is_sender_device = true;
        is_leftmost = true;
    } else {
        // this is a sender device
        is_sender_device = true;
        is_leftmost = false;
    }

    auto semaphore_round1_fw = semaphores[0];
    auto semaphore_round1_bw = semaphores[1];
    auto semaphore_round2_fw = semaphores[2];
    auto semaphore_round2_bw = semaphores[3];
    auto coord_semaphore = semaphores[4];

    // Extract shard grid from input tensor
    TT_FATAL(input_tensor_l.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor_l.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    // Get all cores from the shard grid (data cores)
    std::vector<CoreCoord> all_coord_cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto cores = corerange_to_cores(core_range, std::nullopt);
        all_coord_cores.insert(all_coord_cores.end(), cores.begin(), cores.end());
    }

    // Automate core selection logic
    // Determine mux cores early so we can exclude them from worker core selection
    std::vector<CoreCoord> mux_cores_for_exclusion = {
        CoreCoord(2, 0), CoreCoord(2, 1), CoreCoord(2, 2), CoreCoord(2, 3)};
    if (operation_attributes.input_mux_cores.has_value()) {
        mux_cores_for_exclusion = operation_attributes.input_mux_cores.value();
    }
    CoreRangeSet mux_cores_set = CoreRangeSet(mux_cores_for_exclusion);

    // Get full compute grid and find available cores (excluding shard and mux cores)
    auto compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRangeSet full_grid(CoreRange({0, 0}, {compute_grid.x - 1, compute_grid.y - 1}));
    CoreRangeSet available_cores = full_grid.subtract(shard_grid).subtract(mux_cores_set);

    // Pick 8 cores from available_cores for non_shard_cores
    const uint32_t num_non_shard_cores_needed = all_coord_cores.size();  // Same size as shard grid
    auto available_cores_vec = corerange_to_cores(available_cores, std::nullopt, true);
    TT_FATAL(
        available_cores_vec.size() >= num_non_shard_cores_needed,
        "Not enough available cores for non_shard_cores. Need {} but only {} available.",
        num_non_shard_cores_needed,
        available_cores_vec.size());
    std::vector<CoreCoord> non_shard_cores_vec(
        available_cores_vec.begin(), available_cores_vec.begin() + num_non_shard_cores_needed);
    CoreRangeSet non_shard_grid = CoreRangeSet(non_shard_cores_vec);

    if (operation_attributes.extra_worker_cores.has_value()) {
        non_shard_cores_vec = operation_attributes.extra_worker_cores.value();
        non_shard_grid = CoreRangeSet(non_shard_cores_vec);
    }

    // Construct all_worker_cores: interleave data cores and non-shard cores per link
    // Format: 4 data cores (link 1), 4 non-shard cores (link 1), 4 data cores (link 2), 4 non-shard cores (link 2)
    const uint32_t cores_per_link = all_coord_cores.size() / num_links;
    std::vector<CoreCoord> all_worker_cores_vec;
    all_worker_cores_vec.reserve(all_coord_cores.size() + non_shard_cores_vec.size());

    // Link 1: first half of data cores + first half of non-shard cores
    all_worker_cores_vec.insert(
        all_worker_cores_vec.end(), all_coord_cores.begin(), all_coord_cores.begin() + cores_per_link);
    all_worker_cores_vec.insert(
        all_worker_cores_vec.end(), non_shard_cores_vec.begin(), non_shard_cores_vec.begin() + cores_per_link);

    // Link 2: second half of data cores + second half of non-shard cores
    all_worker_cores_vec.insert(
        all_worker_cores_vec.end(), all_coord_cores.begin() + cores_per_link, all_coord_cores.end());
    all_worker_cores_vec.insert(
        all_worker_cores_vec.end(), non_shard_cores_vec.begin() + cores_per_link, non_shard_cores_vec.end());

    const CoreRangeSet all_cores = CoreRangeSet(all_worker_cores_vec);

    const uint32_t num_shard_cores = all_coord_cores.size();

    uint32_t input_l_total_num_pages = data_movement::get_num_pages(input_tensor_l);
    const uint32_t input_l_num_pages = input_l_total_num_pages / num_shard_cores;
    const uint32_t input_num_tiles = input_l_num_pages;

    const uint32_t input_page_size_bytes = input_tensor_l.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    uint32_t packet_size_bytes = input_num_tiles * input_page_size_bytes;

    tt::tt_metal::Program program{};

    // sdpa compute values
    const auto tile_width = input_tensor_l.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor_l.tensor_spec().tile().get_height();

    bool use_mla = true;
    uint32_t q_heads_parallel_factor = 1;
    uint32_t head_dim_v = input_num_tiles * tile_width;
    // auto q_shape = {1, 1, 8, 512} ; //{1, B, PNH, DH};
    // auto k_shape = {1, 8, 256, 512}; //{B, NKV, S, DH};
    uint32_t PNH = 8;                            // q_shape[2],
    uint32_t DH = input_num_tiles * tile_width;  // k_shape[3];
    uint32_t DHt = DH / tile_width;
    uint32_t vDHt = use_mla ? head_dim_v / tile_width : DHt;
    uint32_t PNHt = PNH / q_heads_parallel_factor / tile_height;

    const uint32_t Sq_chunk_t = PNHt;

    const auto tiny_tile = tt::tt_metal::Tile({8, 32});
    auto stats_tile = tiny_tile;

    // Create buffers
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_l.dtype());

    constexpr auto compute_cb_l = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_compute_l_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{compute_cb_l, input_dataformat}})
            .set_page_size(compute_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_l, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_l_config);

    constexpr auto compute_cb_s = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_compute_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_s, input_dataformat}})
            .set_page_size(compute_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_s, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_s_config);

    constexpr auto compute_cb_m = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_compute_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_m, input_dataformat}})
            .set_page_size(compute_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_m, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_m_config);

    constexpr auto compute_cb_2_l = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_compute_2_l_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{compute_cb_2_l, input_dataformat}})
            .set_page_size(compute_cb_2_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_l, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_2_l_config);

    constexpr auto compute_cb_2_s = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_compute_2_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_s, input_dataformat}})
            .set_page_size(compute_cb_2_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_s, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_2_s_config);

    constexpr auto compute_cb_2_m = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_compute_2_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_m, input_dataformat}})
            .set_page_size(compute_cb_2_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_m, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_2_m_config);

    constexpr auto packet_header_cb_id = tt::CBIndex::c_6;
    constexpr auto buffering_factor = 2;
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_header_config);

    auto total_pkt_size = packet_size_bytes + 1024;
    constexpr auto packet_cb_id = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(2 * total_pkt_size, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, total_pkt_size)
            .set_tile_dims(packet_cb_id, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    // intermediate buffers for compute
    constexpr auto intermediate_cb_l = tt::CBIndex::c_8;
    const uint32_t intermediate_cb_l_size_bytes = input_num_tiles * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_l_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_l_size_bytes, {{intermediate_cb_l, input_dataformat}})
            .set_page_size(intermediate_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_l, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_intermediate_l_config);

    constexpr auto intermediate_cb_s = tt::CBIndex::c_9;
    const uint32_t intermediate_cb_s_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_s_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_s_size_bytes, {{intermediate_cb_s, input_dataformat}})
            .set_page_size(intermediate_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_s, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_intermediate_s_config);

    constexpr auto intermediate_cb_m = tt::CBIndex::c_10;
    const uint32_t intermediate_cb_m_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_m_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_m_size_bytes, {{intermediate_cb_m, input_dataformat}})
            .set_page_size(intermediate_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_m, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_intermediate_m_config);

    constexpr auto compute_out_cb_l = tt::CBIndex::c_11;
    tt::tt_metal::CircularBufferConfig cb_compute_out_l_config =
        tt::tt_metal::CircularBufferConfig(
            input_num_tiles * aligned_input_page_size_bytes, {{compute_out_cb_l, input_dataformat}})
            .set_page_size(compute_out_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_l, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_out_l_config);

    constexpr auto compute_out_cb_s = tt::CBIndex::c_12;
    tt::tt_metal::CircularBufferConfig cb_compute_out_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_s, input_dataformat}})
            .set_page_size(compute_out_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_s, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_out_s_config);

    constexpr auto compute_out_cb_m = tt::CBIndex::c_13;
    tt::tt_metal::CircularBufferConfig cb_compute_out_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_m, input_dataformat}})
            .set_page_size(compute_out_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_m, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_compute_out_m_config);

    constexpr auto cb_exp_max_diff_2 = tt::CBIndex::c_14;
    constexpr auto cb_exp_num_pages = 1;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_2_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff_2, input_dataformat}})
            .set_page_size(cb_exp_max_diff_2, aligned_input_page_size_bytes)
            .set_tile_dims(cb_exp_max_diff_2, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_exp_max_diff_2_config);

    constexpr auto cb_exp_max_diff = tt::CBIndex::c_15;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff, input_dataformat}})
            .set_page_size(cb_exp_max_diff, aligned_input_page_size_bytes)
            .set_tile_dims(cb_exp_max_diff, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_exp_max_diff_config);

    constexpr auto cb_m_temp = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_m_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_m_temp, input_dataformat}})
            .set_page_size(cb_m_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_m_temp, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_m_temp_config);

    constexpr auto packet_header_cb_id_2 = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig cb_header_config_2 =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id_2, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id_2, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id_2, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_header_config_2);

    constexpr auto packet_cb_id_2 = tt::CBIndex::c_18;
    tt::tt_metal::CircularBufferConfig cb_packet_config_2 =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id_2, input_dataformat}})
            .set_page_size(packet_cb_id_2, packet_size_bytes)
            .set_tile_dims(packet_cb_id_2, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_packet_config_2);

    constexpr auto cb_s_temp = tt::CBIndex::c_19;
    tt::tt_metal::CircularBufferConfig cb_s_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s_temp, input_dataformat}})
            .set_page_size(cb_s_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s_temp, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_s_temp_config);

    constexpr auto cb_s1_temp = tt::CBIndex::c_20;
    tt::tt_metal::CircularBufferConfig cb_s1_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s1_temp, input_dataformat}})
            .set_page_size(cb_s1_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s1_temp, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_s1_temp_config);

    constexpr auto cb_s2_temp = tt::CBIndex::c_21;
    tt::tt_metal::CircularBufferConfig cb_s2_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s2_temp, input_dataformat}})
            .set_page_size(cb_s2_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s2_temp, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_s2_temp_config);

    constexpr auto cb_l1_temp = tt::CBIndex::c_22;
    tt::tt_metal::CircularBufferConfig cb_l1_temp_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{cb_l1_temp, input_dataformat}})
            .set_page_size(cb_l1_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_l1_temp, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_l1_temp_config);

    constexpr auto cb_l2_temp = tt::CBIndex::c_23;
    tt::tt_metal::CircularBufferConfig cb_l2_temp_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{cb_l2_temp, input_dataformat}})
            .set_page_size(cb_l2_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_l2_temp, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_l2_temp_config);

    constexpr auto round1_interm_cb_id = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig cb_round1_interm_config =
        tt::tt_metal::CircularBufferConfig(2 * total_pkt_size, {{round1_interm_cb_id, input_dataformat}})
            .set_page_size(round1_interm_cb_id, total_pkt_size)
            .set_tile_dims(round1_interm_cb_id, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_round1_interm_config);

    constexpr auto packet_header_cb_id0 = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig cb_header_config_id0 =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id0, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id0, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id0, stats_tile);
    CreateCircularBuffer(program, all_cores, cb_header_config_id0);

    constexpr auto packet_cb_id0 = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig packet_cb_id0_config =
        tt::tt_metal::CircularBufferConfig(2 * total_pkt_size, {{packet_cb_id0, input_dataformat}})
            .set_page_size(packet_cb_id0, total_pkt_size)
            .set_tile_dims(packet_cb_id0, stats_tile);
    CreateCircularBuffer(program, all_cores, packet_cb_id0_config);

    tt::tt_metal::KernelHandle reader_kernel1 = 0;
    tt::tt_metal::KernelHandle writer_kernel1 = 0;
    tt::tt_metal::KernelHandle reader_kernel2 = 0;
    tt::tt_metal::KernelHandle writer_kernel2 = 0;
    std::vector<uint32_t> reader_ct_args1;
    std::vector<uint32_t> writer_ct_args1;
    std::vector<uint32_t> compute_ct_args;
    std::vector<uint32_t> reader_ct_args2;
    std::vector<uint32_t> writer_ct_args2;

    // 1. Setup muxes for each device type
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const uint32_t num_workers_per_direction = 8;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    std::vector<CoreCoord> mux_cores = {CoreCoord(2, 0), CoreCoord(2, 1), CoreCoord(2, 2), CoreCoord(2, 3)};

    if (operation_attributes.input_mux_cores.has_value()) {
        mux_cores = operation_attributes.input_mux_cores.value();
    }
    auto all_mux_cores = mux_cores;

    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_cores);

    tt::tt_fabric::FabricMuxConfig mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_workers_per_direction, 0, 2, 0, buffer_size_bytes_full_size_channel, mux_base_l1_address);

    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    // scale = 1/sqrt(head_size)
    // Encode scale as float-to-uint32 using union for proper bit representation
    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale_fp32;
    uint32_t scale_val = scale_union.u;
    uint32_t round = 0;

    auto compute_kernel_configuration = ttnn::init_device_compute_kernel_config(
        input_tensor_l.device()->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor_l.device()->arch(), compute_kernel_configuration);

    reader_ct_args1 = {
        input_num_tiles,
        input_page_size_bytes,
        packet_cb_id0,
        0,
        packet_header_cb_id,
        packet_cb_id,
        compute_cb_2_l,
        compute_cb_2_s,
        compute_cb_2_m,
        l1_alignment,
        compute_cb_l,
        compute_cb_s,
        compute_cb_m,
        input_num_tiles,
        input_page_size_bytes,
        packet_size_bytes,
        round1_interm_cb_id};
    reader_ct_args1[3] = reader_ct_args1.size();
    fabric_mux_ct_args(
        num_workers_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        reader_ct_args1);

    writer_ct_args1 = {
        0,
        packet_header_cb_id0,
        packet_cb_id0,
        l1_alignment,
        input_num_tiles,
        input_page_size_bytes,
        packet_size_bytes,
        compute_out_cb_l,
        compute_out_cb_s,
        compute_out_cb_m};
    writer_ct_args1[0] = writer_ct_args1.size();

    fabric_mux_ct_args(
        num_workers_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_ct_args1);

    reader_ct_args2 = {
        0,
        packet_header_cb_id,
        packet_cb_id,
        compute_cb_l,
        compute_cb_s,
        compute_cb_m,
        l1_alignment,
        compute_cb_2_l,
        compute_cb_2_s,
        compute_cb_2_m,
        input_num_tiles,
        input_page_size_bytes,
        packet_size_bytes};
    reader_ct_args2[0] = reader_ct_args2.size();

    fabric_mux_ct_args(
        num_workers_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        reader_ct_args2);

    writer_ct_args2 = {
        0,
        compute_out_cb_l,
        compute_out_cb_s,
        compute_out_cb_m,
        packet_header_cb_id_2,
        packet_cb_id_2,
        l1_alignment,
        input_num_tiles,
        input_page_size_bytes,
        packet_size_bytes};
    writer_ct_args2[0] = writer_ct_args2.size();
    fabric_mux_ct_args(
        num_workers_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_ct_args2);

    compute_ct_args = {
        compute_out_cb_l,
        compute_cb_l,
        compute_cb_2_l,
        compute_cb_s,
        compute_cb_2_m,
        compute_cb_m,
        compute_out_cb_m,
        cb_exp_max_diff_2,
        compute_cb_2_s,
        cb_exp_max_diff,
        compute_out_cb_s,
        cb_m_temp,
        cb_s_temp,
        cb_s1_temp,
        cb_s2_temp,
        cb_l1_temp,
        cb_l2_temp,
        scale_val,
        Sq_chunk_t,
        vDHt,
        round,
        intermediate_cb_l,
        intermediate_cb_s,
        intermediate_cb_m};

    // devices 0 and 2:
    // shard grid: fw: reader and writer 1:
    // non shard grid: bw: reader and writer 2
    if ((is_sender_device && is_leftmost) || is_root2_device) {
        reader_kernel1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/reader1.cpp",
            shard_grid,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args1));

        writer_kernel1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/writer1.cpp",
            shard_grid,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args1));

        reader_kernel2 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/reader2.cpp",
            non_shard_grid,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args2));

        writer_kernel2 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/writer2.cpp",
            non_shard_grid,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args2));

        // handles Round 2 compute, runs with reader1/writer1 on shard_grid
        compute_ct_args[20] = 1;  // round 2
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/compute_kernel.cpp",
            shard_grid,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });

        // handles Round 1 compute, runs with reader2/writer2 on non_shard_grid
        compute_ct_args[20] = 0;
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/compute_kernel.cpp",
            non_shard_grid,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });
    } else {
        // devices 1 and 3:
        // shard grid: fw: reader and writer 2
        // non shard grid: bw: reader and writer 1
        reader_kernel1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/reader1.cpp",
            non_shard_grid,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args1));

        reader_kernel2 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/reader2.cpp",
            shard_grid,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args2));

        writer_kernel1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/writer1.cpp",
            non_shard_grid,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args1));

        writer_kernel2 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/writer2.cpp",
            shard_grid,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args2));

        compute_ct_args[20] = 0;  // round 1
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/compute_kernel.cpp",
            shard_grid,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });

        compute_ct_args[20] = 1;  // round 2
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/compute_kernel.cpp",
            non_shard_grid,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });
    }
    // set runtime args

    // Split cores into links - divide all cores evenly between links
    std::vector<CoreCoord> cores1;
    std::vector<CoreCoord> cores2;
    std::vector<CoreRangeSet> cores_per_link_set;
    std::vector<CoreRangeSet> worker_cores_per_link;

    // Split cores evenly: first half to link 1, second half to link 2
    const uint32_t cores_per_link_count = num_shard_cores / num_links;
    TT_FATAL(
        num_shard_cores % num_links == 0,
        "Number of shard cores ({}) must be evenly divisible by number of links ({})",
        num_shard_cores,
        num_links);

    // data cores per link
    std::vector<CoreCoord> cores_link_1(all_coord_cores.begin(), all_coord_cores.begin() + cores_per_link_count);
    std::vector<CoreCoord> cores_link_2(all_coord_cores.begin() + cores_per_link_count, all_coord_cores.end());
    std::vector<std::vector<CoreCoord>> cores_link_vec = {cores_link_1, cores_link_2};

    cores_per_link_set.push_back(CoreRangeSet(cores_link_1));
    cores_per_link_set.push_back(CoreRangeSet(cores_link_2));

    // worker cores per link
    const uint32_t num_worker_cores = all_worker_cores_vec.size();
    const uint32_t worker_cores_per_link_count = num_worker_cores / num_links;
    std::vector<CoreCoord> worker_cores_link_1(
        all_worker_cores_vec.begin(), all_worker_cores_vec.begin() + worker_cores_per_link_count);
    std::vector<CoreCoord> worker_cores_link_2(
        all_worker_cores_vec.begin() + worker_cores_per_link_count, all_worker_cores_vec.end());

    worker_cores_per_link.push_back(CoreRangeSet(worker_cores_link_1));
    worker_cores_per_link.push_back(CoreRangeSet(worker_cores_link_2));

    // Set termination master to the first core of each link
    // separate termination masters for each mux direction:
    // - backward_mux_term_master[link] = Reader1 will terminate
    // - forward_mux_term_master[link] = Reader2 will terminate
    uint32_t num_worker_cores_per_link_per_dir = 4;
    std::vector<CoreCoord> termination_masters = {
        worker_cores_link_1[0],
        worker_cores_link_2[0],
        worker_cores_link_1[num_worker_cores_per_link_per_dir],
        worker_cores_link_2[num_worker_cores_per_link_per_dir]};

    auto get_bwd_mux_term_master = [&](uint32_t link_idx) { return termination_masters[link_idx]; };
    auto get_fwd_mux_term_master = [&](uint32_t link_idx) { return termination_masters[link_idx + 2]; };

    std::vector<uint32_t> shared_term_sync_sems;
    shared_term_sync_sems.reserve(termination_masters.size());
    for (auto& termination_master : termination_masters) {
        shared_term_sync_sems.push_back(CreateSemaphore(program, {termination_master}, 0));
    }
    auto get_bwd_mux_term_sem = [&](uint32_t link_idx) { return shared_term_sync_sems[link_idx]; };
    auto get_fwd_mux_term_sem = [&](uint32_t link_idx) { return shared_term_sync_sems[link_idx + 2]; };

    std::vector<std::vector<uint32_t>> writer_barrier_sems(num_links);
    std::vector<std::vector<std::vector<CoreCoord>>> writer_barrier_noc_coords(num_links);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        writer_barrier_sems[link_idx].resize(2);  // 2 directions
        writer_barrier_noc_coords[link_idx].resize(2);
        auto& worker_cores_for_link = (link_idx == 0) ? worker_cores_link_1 : worker_cores_link_2;

        for (uint32_t dir = 0; dir < 2; dir++) {
            uint32_t base_idx = dir * num_worker_cores_per_link_per_dir;

            // Collect all cores for this direction
            std::vector<CoreCoord> all_dir_cores;
            for (uint32_t offset = 0; offset < num_worker_cores_per_link_per_dir; offset++) {
                CoreCoord core = worker_cores_for_link[base_idx + offset];
                CoreCoord noc_coord = mesh_device->worker_core_from_logical_core(core);
                writer_barrier_noc_coords[link_idx][dir].push_back(noc_coord);
                all_dir_cores.push_back(core);
            }

            CoreRangeSet all_dir_core_set(all_dir_cores);
            writer_barrier_sems[link_idx][dir] = CreateSemaphore(program, all_dir_core_set, 0);
        }
    }

    auto add_writer_barrier_args =
        [&](uint32_t link_idx, uint32_t dir, uint32_t core_offset, std::vector<uint32_t>& writer_rt_args) {
            bool is_barrier_leader = (core_offset == 0);
            writer_rt_args.push_back(is_barrier_leader ? 1 : 0);

            // All cores use the same shared semaphore address for this direction
            writer_rt_args.push_back(writer_barrier_sems[link_idx][dir]);  // local_barrier_sem_addr

            if (is_barrier_leader) {
                writer_rt_args.push_back(num_worker_cores_per_link_per_dir - 1);  // num_barrier_dests = 3

                // Compute bounding box of non-leader cores for multicast
                uint32_t min_x = writer_barrier_noc_coords[link_idx][dir][1].x;
                uint32_t max_x = writer_barrier_noc_coords[link_idx][dir][1].x;
                uint32_t min_y = writer_barrier_noc_coords[link_idx][dir][1].y;
                uint32_t max_y = writer_barrier_noc_coords[link_idx][dir][1].y;
                for (uint32_t i = 2; i < num_worker_cores_per_link_per_dir; i++) {
                    min_x = std::min(min_x, (uint32_t)writer_barrier_noc_coords[link_idx][dir][i].x);
                    max_x = std::max(max_x, (uint32_t)writer_barrier_noc_coords[link_idx][dir][i].x);
                    min_y = std::min(min_y, (uint32_t)writer_barrier_noc_coords[link_idx][dir][i].y);
                    max_y = std::max(max_y, (uint32_t)writer_barrier_noc_coords[link_idx][dir][i].y);
                }
                writer_rt_args.push_back(min_x);  // mcast_start_x
                writer_rt_args.push_back(min_y);  // mcast_start_y
                writer_rt_args.push_back(max_x);  // mcast_end_x
                writer_rt_args.push_back(max_y);  // mcast_end_y
            } else {
                // Non-leader: num_barrier_dests = 0 and dummy mcast coords
                writer_rt_args.push_back(0);  // num_barrier_dests = 0
                writer_rt_args.push_back(0);  // mcast_start_x (unused)
                writer_rt_args.push_back(0);  // mcast_start_y (unused)
                writer_rt_args.push_back(0);  // mcast_end_x (unused)
                writer_rt_args.push_back(0);  // mcast_end_y (unused)
            }
        };

    uint32_t mux_core_offset = 0;

    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        for (uint32_t dir = 0; dir < 2; dir++) {
            CoreCoord mux_logical_core = all_mux_cores[mux_core_offset++];
            std::vector<uint32_t> mux_rt_args = {};
            const auto src_node_id = mesh_device->get_fabric_node_id(device_coordinate);
            if (dir) {  // forward
                if (forward_coord.has_value()) {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link_idx, program, {mux_logical_core});
                    tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
                }
            } else {
                if (backward_coord.has_value()) {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link_idx, program, {mux_logical_core});
                    tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
                }
            }
        }

        uint32_t worker_id = 0;
        uint32_t core_idx = 0;
        for (auto c : corerange_to_cores(worker_cores_per_link[link_idx], std::nullopt)) {
            std::vector<uint32_t> reader_runtime_args;
            std::vector<uint32_t> writer_runtime_args;

            auto data_core = cores_link_vec[link_idx].at(core_idx % num_worker_cores_per_link_per_dir);
            auto data_core_coord = device->worker_core_from_logical_core(data_core);
            auto core_noc_x = data_core_coord.x;
            auto core_noc_y = data_core_coord.y;

            auto current_core = mesh_device->worker_core_from_logical_core(c);
            auto current_core_x = current_core.x;
            auto current_core_y = current_core.y;

            CoreCoord bwd_mux_term_master = get_bwd_mux_term_master(link_idx);
            CoreCoord fwd_mux_term_master = get_fwd_mux_term_master(link_idx);

            if ((is_sender_device && is_leftmost) || is_root2_device) {
                if (core_idx < num_worker_cores_per_link_per_dir) {
                    // first 4 cores: reader/writer 1
                    // Writer1: sends data in Round1 to fwd neighbor (D0->D1, D2->D3) - uses FORWARD mux
                    // Reader1: receives data in Round2 from bwd neighbor (D3->D0, D1->D2) - uses BACKWARD mux
                    CoreCoord mux_virtual_core_bwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[link_idx * 2]);
                    CoreCoord mux_virtual_core_fwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[(link_idx * 2) + 1]);

                    reader_runtime_args = {
                        input_tensor_l.buffer()->address(),
                        input_tensor_ms.buffer()->address(),  // Combined MS
                        core_noc_x,
                        core_noc_y,
                        current_core_x,
                        current_core_y,
                        fw_intermediate_tensor.buffer()->address(),
                        semaphore_round2_fw.address(),
                        round1_intermediate_tensor.buffer()->address(),
                        coord_semaphore.address()};

                    fabric_mux_rt_args(
                        c == bwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_bwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(bwd_mux_term_master),
                        get_bwd_mux_term_sem(link_idx),
                        reader_runtime_args);
                    reader_runtime_args.push_back(core_idx == 0 ? 1 : 0);  // is_barrier_leader
                    writer_runtime_args = {
                        fw_intermediate_tensor.buffer()->address(),
                        semaphore_round1_fw.address(),
                        core_noc_x,
                        core_noc_y,
                        current_core_x,
                        current_core_y,
                        output_tensor_l.buffer()->address()};  // Only normalized L output
                    // Writer1 uses FORWARD mux (Round1: sends D0->D1, D2->D3)
                    // Use fwd_mux_term_master for forward mux termination coordination
                    fabric_mux_rt_args(
                        c == fwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_fwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(fwd_mux_term_master),
                        get_fwd_mux_term_sem(link_idx),
                        writer_runtime_args);
                    add_writer_barrier_args(link_idx, 0, core_idx, writer_runtime_args);

                    tt::tt_metal::SetRuntimeArgs(program, reader_kernel1, c, reader_runtime_args);
                    tt::tt_metal::SetRuntimeArgs(program, writer_kernel1, c, writer_runtime_args);
                    cores1.push_back(c);
                } else {
                    // second 4 cores: reader/writer 2
                    // Reader2: receives barrier in Round1 from fwd neighbor (D1->D0, D3->D2) - uses FORWARD mux
                    // Writer2: sends data in Round2 to bwd neighbor (D0->D3, D2->D1) - uses BACKWARD mux
                    CoreCoord mux_virtual_core_bwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[link_idx * 2]);
                    CoreCoord mux_virtual_core_fwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[(link_idx * 2) + 1]);

                    // For D0/D2 second 4 cores, Writer2 needs to signal the paired Reader1 on first 4 cores
                    auto& worker_cores_for_link = (link_idx == 0) ? worker_cores_link_1 : worker_cores_link_2;
                    auto paired_reader1_core = worker_cores_for_link[core_idx - num_worker_cores_per_link_per_dir];
                    auto paired_reader1_noc = mesh_device->worker_core_from_logical_core(paired_reader1_core);
                    auto paired_reader1_noc_x = paired_reader1_noc.x;
                    auto paired_reader1_noc_y = paired_reader1_noc.y;

                    reader_runtime_args = {
                        input_tensor_l.buffer()->address(),
                        input_tensor_ms.buffer()->address(),  // Combined MS
                        bw_intermediate_tensor.buffer()->address(),
                        semaphore_round1_bw.address(),
                        core_noc_x,
                        core_noc_y,
                        current_core_x,
                        current_core_y};

                    fabric_mux_rt_args(
                        c == fwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_fwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(fwd_mux_term_master),
                        get_fwd_mux_term_sem(link_idx),
                        reader_runtime_args);
                    reader_runtime_args.push_back(
                        core_idx == num_worker_cores_per_link_per_dir ? 1 : 0);  // is_barrier_leader

                    writer_runtime_args = {
                        bw_intermediate_tensor.buffer()->address(),
                        semaphore_round2_bw.address(),
                        paired_reader1_noc_x,
                        paired_reader1_noc_y,
                        current_core_x,
                        current_core_y,
                        round1_intermediate_tensor.buffer()->address(),
                        coord_semaphore.address(),
                        core_noc_x,
                        core_noc_y,
                    };

                    fabric_mux_rt_args(
                        c == bwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_bwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(bwd_mux_term_master),
                        get_bwd_mux_term_sem(link_idx),
                        writer_runtime_args);
                    add_writer_barrier_args(
                        link_idx, 1, core_idx - num_worker_cores_per_link_per_dir, writer_runtime_args);

                    tt::tt_metal::SetRuntimeArgs(program, reader_kernel2, c, reader_runtime_args);
                    tt::tt_metal::SetRuntimeArgs(program, writer_kernel2, c, writer_runtime_args);
                    cores2.push_back(c);
                }

            } else {
                // devices 1 and 3:
                if (core_idx < num_worker_cores_per_link_per_dir) {
                    // first 4 cores: reader/writer 2
                    CoreCoord mux_virtual_core_bwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[link_idx * 2]);
                    CoreCoord mux_virtual_core_fwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[(link_idx * 2) + 1]);

                    auto& worker_cores_for_link = (link_idx == 0) ? worker_cores_link_1 : worker_cores_link_2;
                    auto paired_reader1_core = worker_cores_for_link[core_idx + num_worker_cores_per_link_per_dir];
                    auto paired_reader1_noc = mesh_device->worker_core_from_logical_core(paired_reader1_core);
                    auto paired_reader1_noc_x = paired_reader1_noc.x;
                    auto paired_reader1_noc_y = paired_reader1_noc.y;

                    reader_runtime_args = {
                        input_tensor_l.buffer()->address(),
                        input_tensor_ms.buffer()->address(),  // Combined MS
                        fw_intermediate_tensor.buffer()->address(),
                        semaphore_round1_fw.address(),
                        core_noc_x,
                        core_noc_y,
                        current_core_x,
                        current_core_y};

                    fabric_mux_rt_args(
                        c == bwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_bwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(bwd_mux_term_master),
                        get_bwd_mux_term_sem(link_idx),
                        reader_runtime_args);
                    reader_runtime_args.push_back(core_idx == 0 ? 1 : 0);  // is_barrier_leader

                    writer_runtime_args = {
                        fw_intermediate_tensor.buffer()->address(),
                        semaphore_round2_fw.address(),
                        paired_reader1_noc_x,
                        paired_reader1_noc_y,
                        current_core_x,
                        current_core_y,
                        round1_intermediate_tensor.buffer()->address(),
                        coord_semaphore.address(),
                        core_noc_x,
                        core_noc_y,
                    };

                    fabric_mux_rt_args(
                        c == fwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_fwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(fwd_mux_term_master),
                        get_fwd_mux_term_sem(link_idx),
                        writer_runtime_args);
                    add_writer_barrier_args(link_idx, 0, core_idx, writer_runtime_args);

                    tt::tt_metal::SetRuntimeArgs(program, reader_kernel2, c, reader_runtime_args);
                    tt::tt_metal::SetRuntimeArgs(program, writer_kernel2, c, writer_runtime_args);
                    cores2.push_back(c);
                } else {
                    // second 4 cores: reader/writer 1
                    CoreCoord mux_virtual_core_bwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[link_idx * 2]);
                    CoreCoord mux_virtual_core_fwd =
                        mesh_device->worker_core_from_logical_core(all_mux_cores[(link_idx * 2) + 1]);

                    reader_runtime_args = {
                        input_tensor_l.buffer()->address(),
                        input_tensor_ms.buffer()->address(),  // Combined MS
                        core_noc_x,
                        core_noc_y,
                        current_core_x,
                        current_core_y,
                        bw_intermediate_tensor.buffer()->address(),
                        semaphore_round2_bw.address(),
                        round1_intermediate_tensor.buffer()->address(),
                        coord_semaphore.address()};

                    fabric_mux_rt_args(
                        c == fwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_fwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(fwd_mux_term_master),
                        get_fwd_mux_term_sem(link_idx),
                        reader_runtime_args);
                    reader_runtime_args.push_back(
                        core_idx == num_worker_cores_per_link_per_dir ? 1 : 0);  // is_barrier_leader
                    writer_runtime_args = {
                        bw_intermediate_tensor.buffer()->address(),
                        semaphore_round1_bw.address(),
                        core_noc_x,
                        core_noc_y,
                        current_core_x,
                        current_core_y,
                        output_tensor_l.buffer()->address()};  // Only normalized L output

                    fabric_mux_rt_args(
                        c == bwd_mux_term_master,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core_bwd,
                        worker_id,
                        c,
                        mux_kernel_config,
                        program,
                        mesh_device->worker_core_from_logical_core(bwd_mux_term_master),
                        get_bwd_mux_term_sem(link_idx),
                        writer_runtime_args);
                    add_writer_barrier_args(
                        link_idx, 1, core_idx - num_worker_cores_per_link_per_dir, writer_runtime_args);

                    tt::tt_metal::SetRuntimeArgs(program, reader_kernel1, c, reader_runtime_args);
                    tt::tt_metal::SetRuntimeArgs(program, writer_kernel1, c, writer_runtime_args);
                    cores1.push_back(c);
                }
            }
            core_idx++;
            worker_id++;
        }
    }

    return {
        std::move(program),
        ReduceToAllOp::ReduceToAll::shared_variables_t{
            .reader_kernel1 = reader_kernel1,
            .reader_kernel2 = reader_kernel2,
            .cores1 = cores1,
            .cores2 = cores2,
            .writer_kernel1 = writer_kernel1,
            .writer_kernel2 = writer_kernel2,
            .semaphores = semaphores,
            .is_device_0_2 = (is_sender_device && is_leftmost) || is_root2_device}};
}

void ReduceToAllOp::ReduceToAll::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /* operation_attributes */,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        // Get the input tensors (combined MS format)
        const auto& input_tensor_l = tensor_args.input_tensor_l;
        const auto& input_tensor_ms = tensor_args.input_tensor_ms;

        // Get output tensors
        const auto& output_tensors_l = tensor_return_value[1];
        const auto& intermediate_tensors = tensor_return_value[0];

        // Handle simplified program with combined MS runtime args layout
        if (shared_variables.is_simplified) {
            for (const auto& core : shared_variables.cores1) {
                // Simplified reader runtime args:
                // 0: R1 neighbor semaphore, 1: R2 neighbor semaphore
                // 2: R1 receive buffer, 3: R2 receive buffer
                auto& reader_runtime_args_by_core =
                    tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel1);
                auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
                reader_runtime_args[0] = shared_variables.semaphores[0].address();     // R1 recv sem
                reader_runtime_args[1] = shared_variables.semaphores[1].address();     // R2 recv sem
                reader_runtime_args[2] = intermediate_tensors[0].buffer()->address();  // R1 recv buffer
                reader_runtime_args[3] = intermediate_tensors[1].buffer()->address();  // R2 recv buffer

                // Simplified writer runtime args layout (combined MS format):
                // 0: input L, 1: input MS
                // 2: R1 mesh_id (static), 3: R1 chip_id (static)
                // 4: R1 dest addr, 5: R1 sem addr
                // 6: R2 mesh_id (static), 7: R2 chip_id (static)
                // 8: R2 dest addr, 9: R2 sem addr
                // 10-19: aggregator args (static)
                auto& writer_runtime_args_by_core =
                    tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel1);
                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
                writer_runtime_args[0] = input_tensor_l.buffer()->address();
                writer_runtime_args[1] = input_tensor_ms.buffer()->address();
                // Indices 2-3 (mesh_id, chip_id) are static - don't update
                writer_runtime_args[4] = intermediate_tensors[0].buffer()->address();  // R1 dest
                writer_runtime_args[5] = shared_variables.semaphores[0].address();     // R1 sem
                // Indices 6-7 (mesh_id, chip_id) are static - don't update
                writer_runtime_args[8] = intermediate_tensors[1].buffer()->address();  // R2 dest
                writer_runtime_args[9] = shared_variables.semaphores[1].address();     // R2 sem
            }

            // Update CB addresses for aliased buffers (critical for trace replay)
            // When trace replays, the CBs still point to old buffer addresses unless updated
            if (shared_variables.cb_local_l_handle.has_value()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program,
                    shared_variables.cb_local_l_handle.value(),
                    *input_tensor_l.buffer(),
                    shared_variables.l_tile_size);
            }
            if (shared_variables.cb_local_ms_handle.has_value()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program,
                    shared_variables.cb_local_ms_handle.value(),
                    *input_tensor_ms.buffer(),
                    shared_variables.ms_tile_size);
            }
            if (shared_variables.cb_r1_neighbor_l_handle.has_value()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program,
                    shared_variables.cb_r1_neighbor_l_handle.value(),
                    *intermediate_tensors[0].buffer(),
                    shared_variables.l_tile_size);
            }
            if (shared_variables.cb_r2_neighbor_l_handle.has_value()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program,
                    shared_variables.cb_r2_neighbor_l_handle.value(),
                    *intermediate_tensors[1].buffer(),
                    shared_variables.l_tile_size);
            }
            if (shared_variables.cb_l_out_handle.has_value()) {
                UpdateDynamicCircularBufferAddressAndTotalSize(
                    program,
                    shared_variables.cb_l_out_handle.value(),
                    *output_tensors_l[0].buffer(),
                    shared_variables.l_tile_size);
            }

            continue;  // Skip original program logic
        }

        for (const auto& core : shared_variables.cores1) {
            // Update reader1 runtime args (old program - deprecated but keeping for compatibility)
            auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel1);
            auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
            reader_runtime_args[0] = input_tensor_l.buffer()->address();
            reader_runtime_args[1] = input_tensor_ms.buffer()->address();  // Combined MS
            // D0/D2 uses fw_intermediate, D1/D3 uses bw_intermediate
            reader_runtime_args[7] = shared_variables.is_device_0_2 ? intermediate_tensors[0].buffer()->address()
                                                                    : intermediate_tensors[1].buffer()->address();
            // D0/D2 uses semaphore_round2_fw , D1/D3 uses semaphore_round2_bw
            reader_runtime_args[8] = shared_variables.is_device_0_2 ? shared_variables.semaphores[2].address()
                                                                    : shared_variables.semaphores[3].address();
            reader_runtime_args[9] = intermediate_tensors[2].buffer()->address();
            reader_runtime_args[10] = shared_variables.semaphores[4].address();

            // Update writer1 runtime args
            auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel1);
            auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
            // D0/D2 uses fw_intermediate, D1/D3 uses bw_intermediate
            writer_runtime_args[0] = shared_variables.is_device_0_2 ? intermediate_tensors[0].buffer()->address()
                                                                    : intermediate_tensors[1].buffer()->address();
            // D0/D2 uses semaphore_round1_fw, D1/D3 uses semaphore_round1_bw
            writer_runtime_args[1] = shared_variables.is_device_0_2 ? shared_variables.semaphores[0].address()
                                                                    : shared_variables.semaphores[1].address();
            writer_runtime_args[6] = output_tensors_l[0].buffer()->address();
        }

        for (const auto& core : shared_variables.cores2) {
            // Update reader2 runtime args (old program - deprecated but keeping for compatibility)
            auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel2);
            auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
            reader_runtime_args[0] = input_tensor_l.buffer()->address();
            reader_runtime_args[1] = input_tensor_ms.buffer()->address();  // Combined MS
            // D0/D2 uses bw_intermediate, D1/D3 uses fw_intermediate
            reader_runtime_args[3] = shared_variables.is_device_0_2 ? intermediate_tensors[1].buffer()->address()
                                                                    : intermediate_tensors[0].buffer()->address();
            // D0/D2 uses semaphore_round1_bw, D1/D3 uses semaphore_round1_fw
            reader_runtime_args[4] = shared_variables.is_device_0_2 ? shared_variables.semaphores[1].address()
                                                                    : shared_variables.semaphores[0].address();

            // Update writer2 runtime args
            auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel2);
            auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
            // D0/D2 uses bw_intermediate, D1/D3 uses fw_intermediate
            writer_runtime_args[0] = shared_variables.is_device_0_2 ? intermediate_tensors[1].buffer()->address()
                                                                    : intermediate_tensors[0].buffer()->address();
            // D0/D2 uses semaphore_round2_bw, D1/D3 uses semaphore_round2_fw
            writer_runtime_args[1] = shared_variables.is_device_0_2 ? shared_variables.semaphores[3].address()
                                                                    : shared_variables.semaphores[2].address();
            writer_runtime_args[6] = intermediate_tensors[2].buffer()->address();
            writer_runtime_args[7] = shared_variables.semaphores[4].address();
        }
    }
};

}  // namespace ttnn::operations::ccl

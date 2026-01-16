// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "reduce_to_root_op.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::ccl {

inline bool mux_connection_valid(uint32_t dir, bool is_leftmost, bool is_sender_device) {
    if (is_sender_device) {
        return (dir && !is_leftmost) || (!dir && is_leftmost);
    }
    return true;
};

inline std::vector<MeshCoordinate> find_send_recv(
    uint32_t dir,
    bool is_leftmost,
    bool is_sender_device,
    bool is_root_device,
    bool is_root2_device,
    const MeshCoordinate& device_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord) {
    std::vector<MeshCoordinate> transfer_coords;
    MeshCoordinate send_coord = device_coordinate;
    MeshCoordinate receive_coord = device_coordinate;
    if (is_sender_device) {
        if (is_leftmost) {  // left
            if (forward_coord.has_value() == 0) {
                TT_FATAL(false, "ReduceToRoot: leftmost sender device must have a forward coordinate defined.");
            }
            send_coord = device_coordinate;
            receive_coord = forward_coord.has_value() ? forward_coord.value() : device_coordinate;
        } else {  // right
            if (backward_coord.has_value() == 0) {
                TT_FATAL(false, "ReduceToRoot: rightmost sender device must have a backward coordinate defined.");
            }
            send_coord = device_coordinate;
            receive_coord = backward_coord.has_value() ? backward_coord.value() : device_coordinate;
        }
    } else if (is_root_device) {
        if (backward_coord.has_value() == 0 || forward_coord.has_value() == 0) {
            TT_FATAL(false, "ReduceToRoot: root sender device must have a fwd and backward coordinate defined.");
        }
        // switch send and recv when the device acts like a receiver
        send_coord = device_coordinate;
        auto sender_coord_1 = backward_coord.has_value() ? backward_coord.value() : device_coordinate;
        auto sender_coord_2 = forward_coord.has_value() ? forward_coord.value() : device_coordinate;
        if (dir == 0) {
            receive_coord = sender_coord_1;
        } else {
            receive_coord = sender_coord_2;
        }
    } else if (is_root2_device) {
        if (backward_coord.has_value() == 0 || forward_coord.has_value() == 0) {
            TT_FATAL(false, "ReduceToRoot: root2 sender device must have a fwd and backward coordinate defined.");
        }
        send_coord = device_coordinate;
        receive_coord = forward_coord.has_value() ? forward_coord.value() : device_coordinate;
        if (dir) {
            send_coord = device_coordinate;
            receive_coord = backward_coord.has_value() ? backward_coord.value() : device_coordinate;
        }
    }
    transfer_coords.push_back(send_coord);
    transfer_coords.push_back(receive_coord);
    return transfer_coords;
}

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
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // termination_sync_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);                    // termination_master_noc_x
    worker_rt_args.push_back(termination_master_virtual_core.y);                    // termination_master_noc_y
}

ttnn::device_operation::CachedProgram<ReduceToRootOp::ReduceToRoot::shared_variables_t> reduce_to_root_program_factory(
    const ReduceToRootOp::tensor_args_t& tensor_args,
    const ReduceToRootOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& root_coord,
    const float scale_fp32,
    const MeshCoordinate& device_coordinate,
    std::optional<ttnn::MeshCoordinate>& forward_coord,
    std::optional<ttnn::MeshCoordinate>& backward_coord,
    ReduceToRootOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor_l.device());
    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_s = tensor_args.input_tensor_s;
    const auto& input_tensor_m = tensor_args.input_tensor_m;
    const auto& intermediate_tensor = output_tensors.at(0)[0];

    const auto& output_tensor_l = output_tensors.at(1)[0];
    const auto& output_tensor_s = output_tensors.at(1)[1];
    const auto& output_tensor_m = output_tensors.at(1)[2];

    // check which device within the column:
    // root device
    // root2 device: the intermediate device (has forward and backward neighbors) but not root
    // senders are leaf devices (only have one neighbor)

    // check which device is this one based on coordinates
    auto* device = input_tensor_l.device();
    auto mesh_shape = mesh_device->shape();
    bool is_root_device = false;
    bool is_root2_device = false;
    bool is_sender_device = false;
    bool is_leftmost = false;
    if (device_coordinate == root_coord) {
        // this is the root device
        is_root_device = true;
    } else if (device_coordinate != root_coord && backward_coord.has_value() && backward_coord.value() == root_coord) {
        // this is the intermediate device
        is_root2_device = true;
    } else if (backward_coord.has_value() && forward_coord.has_value() == 0) {
        // this is a sender device
        is_sender_device = true;
        is_leftmost = false;
    } else if (forward_coord.has_value() && backward_coord.has_value() == 0 && forward_coord.value() == root_coord) {
        // this is a sender device
        is_sender_device = true;
        is_leftmost = true;
    }

    auto semaphore_round1 = semaphores[0];
    auto semaphore_round2 = semaphores[1];

    // Extract shard grid from input tensor
    TT_FATAL(input_tensor_l.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor_l.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    // Get all cores from the shard grid
    std::vector<CoreCoord> all_coord_cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto cores = corerange_to_cores(core_range, std::nullopt);
        all_coord_cores.insert(all_coord_cores.end(), cores.begin(), cores.end());
    }
    const CoreRangeSet all_cores = shard_grid;
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
    uint32_t PNH = 8;  // q_shape[2],
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

    constexpr auto compute_cb_s = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_compute_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_s, input_dataformat}})
            .set_page_size(compute_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_s, stats_tile);

    constexpr auto compute_cb_m = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_compute_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_m, input_dataformat}})
            .set_page_size(compute_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_m, stats_tile);

    constexpr auto compute_cb_2_l = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_compute_2_l_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{compute_cb_2_l, input_dataformat}})
            .set_page_size(compute_cb_2_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_l, stats_tile);

    constexpr auto compute_cb_2_s = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_compute_2_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_s, input_dataformat}})
            .set_page_size(compute_cb_2_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_s, stats_tile);

    constexpr auto compute_cb_2_m = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_compute_2_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_m, input_dataformat}})
            .set_page_size(compute_cb_2_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_m, stats_tile);

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

    auto total_pkt_size = packet_size_bytes + 1024;
    constexpr auto packet_cb_id = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(2 * total_pkt_size, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, total_pkt_size)
            .set_tile_dims(packet_cb_id, stats_tile);

    // intermediate buffers for compute
    constexpr auto intermediate_cb_l = tt::CBIndex::c_8;
    const uint32_t intermediate_cb_l_size_bytes = input_num_tiles * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_l_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_l_size_bytes, {{intermediate_cb_l, input_dataformat}})
            .set_page_size(intermediate_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_l, stats_tile);

    constexpr auto intermediate_cb_s = tt::CBIndex::c_9;
    const uint32_t intermediate_cb_s_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_s_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_s_size_bytes, {{intermediate_cb_s, input_dataformat}})
            .set_page_size(intermediate_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_s, stats_tile);

    constexpr auto intermediate_cb_m = tt::CBIndex::c_10;
    const uint32_t intermediate_cb_m_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_m_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_m_size_bytes, {{intermediate_cb_m, input_dataformat}})
            .set_page_size(intermediate_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_m, stats_tile);

    constexpr auto compute_out_cb_l = tt::CBIndex::c_11;
    tt::tt_metal::CircularBufferConfig cb_compute_out_l_config =
        tt::tt_metal::CircularBufferConfig(
            input_num_tiles * aligned_input_page_size_bytes, {{compute_out_cb_l, input_dataformat}})
            .set_page_size(compute_out_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_l, stats_tile);

    constexpr auto compute_out_cb_s = tt::CBIndex::c_12;
    tt::tt_metal::CircularBufferConfig cb_compute_out_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_s, input_dataformat}})
            .set_page_size(compute_out_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_s, stats_tile);

    constexpr auto compute_out_cb_m = tt::CBIndex::c_13;
    tt::tt_metal::CircularBufferConfig cb_compute_out_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_m, input_dataformat}})
            .set_page_size(compute_out_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_m, stats_tile);

    constexpr auto cb_exp_max_diff_2 = tt::CBIndex::c_14;
    constexpr auto cb_exp_num_pages = 1;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_2_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff_2, input_dataformat}})
            .set_page_size(cb_exp_max_diff_2, aligned_input_page_size_bytes)
            .set_tile_dims(cb_exp_max_diff_2, stats_tile);

    constexpr auto cb_exp_max_diff = tt::CBIndex::c_15;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff, input_dataformat}})
            .set_page_size(cb_exp_max_diff, aligned_input_page_size_bytes)
            .set_tile_dims(cb_exp_max_diff, stats_tile);

    constexpr auto cb_m_temp = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_m_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_m_temp, input_dataformat}})
            .set_page_size(cb_m_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_m_temp, stats_tile);

    constexpr auto packet_header_cb_id_2 = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig cb_header_config_2 =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id_2, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id_2, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id_2, stats_tile);

    constexpr auto packet_cb_id_2 = tt::CBIndex::c_18;
    tt::tt_metal::CircularBufferConfig cb_packet_config_2 =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id_2, input_dataformat}})
            .set_page_size(packet_cb_id_2, packet_size_bytes)
            .set_tile_dims(packet_cb_id_2, stats_tile);

    constexpr auto cb_s_temp = tt::CBIndex::c_19;
    tt::tt_metal::CircularBufferConfig cb_s_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s_temp, input_dataformat}})
            .set_page_size(cb_s_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s_temp, stats_tile);

    constexpr auto cb_s1_temp = tt::CBIndex::c_20;
    tt::tt_metal::CircularBufferConfig cb_s1_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s1_temp, input_dataformat}})
            .set_page_size(cb_s1_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s1_temp, stats_tile);

    constexpr auto cb_s2_temp = tt::CBIndex::c_21;
    tt::tt_metal::CircularBufferConfig cb_s2_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s2_temp, input_dataformat}})
            .set_page_size(cb_s2_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s2_temp, stats_tile);

    constexpr auto cb_l1_temp = tt::CBIndex::c_22;
    tt::tt_metal::CircularBufferConfig cb_l1_temp_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{cb_l1_temp, input_dataformat}})
            .set_page_size(cb_l1_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_l1_temp, stats_tile);

    constexpr auto cb_l2_temp = tt::CBIndex::c_23;
    tt::tt_metal::CircularBufferConfig cb_l2_temp_config =
        tt::tt_metal::CircularBufferConfig(
            1 * input_num_tiles * aligned_input_page_size_bytes, {{cb_l2_temp, input_dataformat}})
            .set_page_size(cb_l2_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_l2_temp, stats_tile);

    // create cbs only on needed devices
    if (is_sender_device) {
        CreateCircularBuffer(program, all_cores, cb_header_config);
        CreateCircularBuffer(program, all_cores, cb_packet_config);

    } else if (is_root_device) {
        CreateCircularBuffer(program, all_cores, cb_compute_l_config);
        CreateCircularBuffer(program, all_cores, cb_compute_m_config);
        CreateCircularBuffer(program, all_cores, cb_compute_s_config);
        CreateCircularBuffer(program, all_cores, cb_compute_2_l_config);
        CreateCircularBuffer(program, all_cores, cb_compute_2_m_config);
        CreateCircularBuffer(program, all_cores, cb_compute_2_s_config);
        CreateCircularBuffer(program, all_cores, cb_header_config);
        CreateCircularBuffer(program, all_cores, cb_packet_config);
        CreateCircularBuffer(program, all_cores, cb_intermediate_l_config);
        CreateCircularBuffer(program, all_cores, cb_intermediate_s_config);
        CreateCircularBuffer(program, all_cores, cb_intermediate_m_config);
        CreateCircularBuffer(program, all_cores, cb_compute_out_l_config);
        CreateCircularBuffer(program, all_cores, cb_compute_out_s_config);
        CreateCircularBuffer(program, all_cores, cb_compute_out_m_config);
        CreateCircularBuffer(program, all_cores, cb_exp_max_diff_2_config);
        CreateCircularBuffer(program, all_cores, cb_exp_max_diff_config);
        CreateCircularBuffer(program, all_cores, cb_m_temp_config);
        CreateCircularBuffer(program, all_cores, cb_s_temp_config);
        CreateCircularBuffer(program, all_cores, cb_s1_temp_config);
        CreateCircularBuffer(program, all_cores, cb_s2_temp_config);
        CreateCircularBuffer(program, all_cores, cb_l1_temp_config);
        CreateCircularBuffer(program, all_cores, cb_l2_temp_config);

    } else if (is_root2_device) {
        CreateCircularBuffer(program, all_cores, cb_compute_l_config);
        CreateCircularBuffer(program, all_cores, cb_compute_m_config);
        CreateCircularBuffer(program, all_cores, cb_compute_s_config);
        CreateCircularBuffer(program, all_cores, cb_compute_2_l_config);
        CreateCircularBuffer(program, all_cores, cb_compute_2_m_config);
        CreateCircularBuffer(program, all_cores, cb_compute_2_s_config);
        CreateCircularBuffer(program, all_cores, cb_header_config);
        CreateCircularBuffer(program, all_cores, cb_packet_config);
        CreateCircularBuffer(program, all_cores, cb_header_config_2);
        CreateCircularBuffer(program, all_cores, cb_packet_config_2);
        CreateCircularBuffer(program, all_cores, cb_compute_out_l_config);
        CreateCircularBuffer(program, all_cores, cb_compute_out_s_config);
        CreateCircularBuffer(program, all_cores, cb_compute_out_m_config);
        CreateCircularBuffer(program, all_cores, cb_exp_max_diff_2_config);
        CreateCircularBuffer(program, all_cores, cb_exp_max_diff_config);
        CreateCircularBuffer(program, all_cores, cb_m_temp_config);
        CreateCircularBuffer(program, all_cores, cb_s_temp_config);
        CreateCircularBuffer(program, all_cores, cb_s1_temp_config);
        CreateCircularBuffer(program, all_cores, cb_s2_temp_config);
        CreateCircularBuffer(program, all_cores, cb_l1_temp_config);
        CreateCircularBuffer(program, all_cores, cb_l2_temp_config);
    }

    tt::tt_metal::KernelHandle reader_kernel = 0;
    tt::tt_metal::KernelHandle writer_kernel = 0;
    std::vector<uint32_t> reader_ct_args;
    std::vector<uint32_t> writer_ct_args;
    std::vector<uint32_t> compute_ct_args;

    // 1. Setup muxes for each device type
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const uint32_t num_workers_per_direction = 4;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    std::vector<CoreCoord> mux_cores = {
        CoreCoord(2, 0), CoreCoord(2, 1), CoreCoord(2, 2), CoreCoord(2, 3)};  // to be modified based on device type

    if (operation_attributes.input_mux_cores.has_value()) {
        mux_cores = operation_attributes.input_mux_cores.value();
    }
    auto all_mux_cores = mux_cores;
    if (is_sender_device) {
        mux_cores = {mux_cores[0], mux_cores[2]};
    }

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

    if (is_sender_device) {
        reader_ct_args = {input_num_tiles, input_page_size_bytes, packet_cb_id};
        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/sender_reader_kernel.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

        writer_ct_args = {
            0,
            packet_header_cb_id,
            packet_cb_id,
            l1_alignment,
            input_num_tiles,
            input_page_size_bytes,
            packet_size_bytes};
        writer_ct_args[0] = writer_ct_args.size();

        fabric_mux_ct_args(
            num_workers_per_direction,
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            mux_kernel_config,
            writer_ct_args);

        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/sender_writer_kernel.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    }

    else if (is_root_device) {
        reader_ct_args = {
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
            packet_size_bytes};
        reader_ct_args[0] = reader_ct_args.size();

        fabric_mux_ct_args(
            num_workers_per_direction,
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            mux_kernel_config,
            reader_ct_args);

        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_reader_kernel.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

        writer_ct_args = {compute_out_cb_l, compute_out_cb_s, compute_out_cb_m, input_num_tiles, input_page_size_bytes};
        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_writer_kernel.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    } else if (is_root2_device) {
        reader_ct_args = {
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
        reader_ct_args[0] = reader_ct_args.size();

        fabric_mux_ct_args(
            num_workers_per_direction,
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            mux_kernel_config,
            reader_ct_args);

        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root2_receive_reader_kernel.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

        writer_ct_args = {
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
        writer_ct_args[0] = writer_ct_args.size();

        fabric_mux_ct_args(
            num_workers_per_direction,
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            mux_kernel_config,
            writer_ct_args);

        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root2_writer_kernel.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    }

    if (!is_sender_device) {
        // scale = 1/sqrt(head_size)
        // Encode scale as float-to-uint32 using union for proper bit representation
        union {
            float f;
            uint32_t u;
        } scale_union{};
        scale_union.f = scale_fp32;
        uint32_t scale_val = scale_union.u;
        uint32_t loop_size = is_root_device ? 2 : 1;

        auto compute_kernel_configuration = ttnn::init_device_compute_kernel_config(
            input_tensor_l.device()->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false);

        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input_tensor_l.device()->arch(), compute_kernel_configuration);

        compute_ct_args = {compute_out_cb_l, compute_cb_l,      compute_cb_2_l,    compute_cb_s,     compute_cb_2_m,
                           compute_cb_m,     compute_out_cb_m,  cb_exp_max_diff_2, compute_cb_2_s,   cb_exp_max_diff,
                           compute_out_cb_s, cb_m_temp,         cb_s_temp,         cb_s1_temp,       cb_s2_temp,
                           cb_l1_temp,       cb_l2_temp,        scale_val,         Sq_chunk_t,       vDHt,
                           loop_size,        intermediate_cb_l, intermediate_cb_s, intermediate_cb_m};
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/compute_kernel.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });
    }
    // set runtime args

    // Split cores into links - divide all cores evenly between links
    constexpr auto num_links = 2;
    std::vector<CoreCoord> cores;
    std::vector<CoreRangeSet> cores_per_link;

    // Split cores evenly: first half to link 1, second half to link 2
    const uint32_t cores_per_link_count = num_shard_cores / num_links;
    TT_FATAL(
        num_shard_cores % num_links == 0,
        "Number of shard cores ({}) must be evenly divisible by number of links ({})",
        num_shard_cores,
        num_links);

    std::vector<CoreCoord> cores_link_1(all_coord_cores.begin(), all_coord_cores.begin() + cores_per_link_count);
    std::vector<CoreCoord> cores_link_2(all_coord_cores.begin() + cores_per_link_count, all_coord_cores.end());

    cores_per_link.push_back(CoreRangeSet(cores_link_1));
    cores_per_link.push_back(CoreRangeSet(cores_link_2));

    // Set termination master to the first core of each link
    std::vector<CoreCoord> termination_masters = {cores_link_1[0], cores_link_2[0]};
    CoreCoord termination_master = termination_masters[0];

    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        uint32_t start_idx = link_idx == 0 ? 0 : 2;
        termination_master = termination_masters[link_idx];
        for (uint32_t dir = 0; dir < 2; dir++) {
            CoreCoord mux_logical_core = dir == 0 ? all_mux_cores[start_idx] : all_mux_cores[start_idx + 1];
            if (is_sender_device) {
                mux_logical_core = link_idx == 0 ? all_mux_cores[0] : all_mux_cores[2];
            }
            if (mux_connection_valid(dir, is_leftmost, is_sender_device)) {
                std::vector<uint32_t> mux_rt_args = {};
                auto transfer_coords = find_send_recv(
                    dir,
                    is_leftmost,
                    is_sender_device,
                    is_root_device,
                    is_root2_device,
                    device_coordinate,
                    forward_coord,
                    backward_coord);
                const auto& send_coord = transfer_coords[0];
                const auto& receive_coord = transfer_coords[1];
                const auto src_node_id = mesh_device->get_fabric_node_id(send_coord);
                const auto dst_node_id = mesh_device->get_fabric_node_id(receive_coord);
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link_idx, program, {mux_logical_core});
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }
        }

        uint32_t worker_id = 0;
        for (auto c : corerange_to_cores(cores_per_link[link_idx], std::nullopt)) {
            std::vector<uint32_t> reader_runtime_args;
            std::vector<uint32_t> writer_runtime_args;

            auto data_core_coord = device->worker_core_from_logical_core(c);
            auto core_noc_x = data_core_coord.x;
            auto core_noc_y = data_core_coord.y;

            if (is_sender_device) {
                reader_runtime_args = {
                    input_tensor_l.buffer()->address(),
                    input_tensor_s.buffer()->address(),
                    input_tensor_m.buffer()->address(),
                    core_noc_x,
                    core_noc_y};
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);

                writer_runtime_args = {
                    intermediate_tensor.buffer()->address(),
                    semaphore_round1.address(),
                    core_noc_x,
                    core_noc_y,
                };
                CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(all_mux_cores[start_idx]);
                fabric_mux_rt_args(
                    c == termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core,
                    worker_id,
                    c,
                    mux_kernel_config,
                    program,
                    mesh_device->worker_core_from_logical_core(termination_master),
                    writer_runtime_args);
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel, c, writer_runtime_args);

            } else if (is_root_device) {
                CoreCoord mux_virtual_core_fwd = mesh_device->worker_core_from_logical_core(all_mux_cores[start_idx]);
                CoreCoord mux_virtual_core_bwd =
                    mesh_device->worker_core_from_logical_core(all_mux_cores[start_idx + 1]);
                reader_runtime_args = {
                    0,  // fabric_2_idx,
                    input_tensor_l.buffer()->address(),
                    input_tensor_s.buffer()->address(),
                    input_tensor_m.buffer()->address(),
                    intermediate_tensor.buffer()->address(),
                    semaphore_round1.address(),
                    semaphore_round2.address(),
                    core_noc_x,
                    core_noc_y};

                // first receiving from device on the left
                fabric_mux_rt_args(
                    c == termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_fwd,
                    worker_id,
                    c,
                    mux_kernel_config,
                    program,
                    mesh_device->worker_core_from_logical_core(termination_master),
                    reader_runtime_args);

                reader_runtime_args[0] = reader_runtime_args.size();

                // then receiving from device on the right
                fabric_mux_rt_args(
                    c == termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_bwd,
                    worker_id,
                    c,
                    mux_kernel_config,
                    program,
                    mesh_device->worker_core_from_logical_core(termination_master),
                    reader_runtime_args);

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);
                writer_runtime_args = {
                    output_tensor_l.buffer()->address(),
                    output_tensor_s.buffer()->address(),
                    output_tensor_m.buffer()->address(),
                    core_noc_x,
                    core_noc_y};
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel, c, writer_runtime_args);
            } else if (is_root2_device) {
                CoreCoord mux_virtual_core_fwd = mesh_device->worker_core_from_logical_core(all_mux_cores[start_idx]);
                CoreCoord mux_virtual_core_bwd =
                    mesh_device->worker_core_from_logical_core(all_mux_cores[start_idx + 1]);
                reader_runtime_args = {
                    input_tensor_l.buffer()->address(),
                    input_tensor_s.buffer()->address(),
                    input_tensor_m.buffer()->address(),
                    intermediate_tensor.buffer()->address(),
                    semaphore_round1.address(),
                    core_noc_x,
                    core_noc_y};

                fabric_mux_rt_args(
                    c == termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_fwd,
                    worker_id,
                    c,
                    mux_kernel_config,
                    program,
                    mesh_device->worker_core_from_logical_core(termination_master),
                    reader_runtime_args);

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);
                writer_runtime_args = {
                    intermediate_tensor.buffer()->address(), semaphore_round2.address(), core_noc_x, core_noc_y};
                fabric_mux_rt_args(
                    c == termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_bwd,
                    worker_id,
                    c,
                    mux_kernel_config,
                    program,
                    mesh_device->worker_core_from_logical_core(termination_master),
                    writer_runtime_args);

                tt::tt_metal::SetRuntimeArgs(program, writer_kernel, c, writer_runtime_args);
            }
            cores.push_back(c);
            worker_id++;
        }
    }

    return {
        std::move(program),
        ReduceToRootOp::ReduceToRoot::shared_variables_t{
            .send_unary_reader_kernel_id = is_sender_device ? reader_kernel : 0,
            .send_unary_writer_kernel_id = is_sender_device ? writer_kernel : 0,
            .cores = cores,
            .root1_reader_kernel_id = is_root_device ? reader_kernel : 0,
            .root1_writer_kernel_id = is_root_device ? writer_kernel : 0,
            .root2_reader_kernel_id = is_root2_device ? reader_kernel : 0,
            .root2_writer_kernel_id = is_root2_device ? writer_kernel : 0,
            .semaphores = semaphores}};
}

void ReduceToRootOp::ReduceToRoot::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        // Get the input tensors
        const auto& input_tensor_l = tensor_args.input_tensor_l;
        const auto& input_tensor_s = tensor_args.input_tensor_s;
        const auto& input_tensor_m = tensor_args.input_tensor_m;

        // Get output tensors
        const auto& output_tensors_l = tensor_return_value[1];
        const auto& intermediate_tensors_l = tensor_return_value[0];

        // Determine device type based on which kernels are present
        bool is_sender_device = shared_variables.send_unary_reader_kernel_id != 0;
        bool is_root_device = shared_variables.root1_reader_kernel_id != 0;
        bool is_root2_device = shared_variables.root2_reader_kernel_id != 0;

        // Update sender device runtime args
        if (is_sender_device) {
            auto& reader_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.send_unary_reader_kernel_id);
            auto& writer_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.send_unary_writer_kernel_id);

            for (const auto& core : shared_variables.cores) {
                // Update reader runtime args - input tensor addresses
                auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
                reader_runtime_args[0] = input_tensor_l.buffer()->address();
                reader_runtime_args[1] = input_tensor_s.buffer()->address();
                reader_runtime_args[2] = input_tensor_m.buffer()->address();

                // Update writer runtime args - intermediate tensor address and semaphore
                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
                writer_runtime_args[0] = intermediate_tensors_l[0].buffer()->address();
                writer_runtime_args[1] = shared_variables.semaphores[0].address();
            }
        }

        // Update root device runtime args
        if (is_root_device) {
            auto& reader_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.root1_reader_kernel_id);
            auto& writer_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.root1_writer_kernel_id);

            for (const auto& core : shared_variables.cores) {
                // Update reader runtime args
                auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
                reader_runtime_args[1] = input_tensor_l.buffer()->address();
                reader_runtime_args[2] = input_tensor_s.buffer()->address();
                reader_runtime_args[3] = input_tensor_m.buffer()->address();
                reader_runtime_args[4] = intermediate_tensors_l[0].buffer()->address();
                reader_runtime_args[5] = shared_variables.semaphores[0].address();
                reader_runtime_args[6] = shared_variables.semaphores[1].address();

                // Update writer runtime args - output tensor addresses
                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
                writer_runtime_args[0] = output_tensors_l[0].buffer()->address();
                writer_runtime_args[1] = output_tensors_l[1].buffer()->address();
                writer_runtime_args[2] = output_tensors_l[2].buffer()->address();
            }
        }

        // Update root2 device runtime args
        if (is_root2_device) {
            auto& reader_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.root2_reader_kernel_id);
            auto& writer_runtime_args_by_core =
                tt::tt_metal::GetRuntimeArgs(program, shared_variables.root2_writer_kernel_id);

            for (const auto& core : shared_variables.cores) {
                // Update reader runtime args
                auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
                reader_runtime_args[0] = input_tensor_l.buffer()->address();
                reader_runtime_args[1] = input_tensor_s.buffer()->address();
                reader_runtime_args[2] = input_tensor_m.buffer()->address();
                reader_runtime_args[3] = intermediate_tensors_l[0].buffer()->address();
                reader_runtime_args[4] = shared_variables.semaphores[0].address();

                // Update writer runtime args - intermediate and output tensor addresses
                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
                writer_runtime_args[0] = intermediate_tensors_l[0].buffer()->address();
                writer_runtime_args[1] = shared_variables.semaphores[1].address();
            }
        }
    }
};
}  // namespace ttnn::operations::ccl

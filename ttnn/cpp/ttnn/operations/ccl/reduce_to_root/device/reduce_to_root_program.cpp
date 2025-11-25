// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "reduce_to_root_op.hpp"

// ASSUMING ROOT ID ALWAYS DEVICE 1
// CHANGE HARDCODED VALUES IF DEVICE 2 IS ROOT INSTEAD

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
    const MeshCoordinate& device_coordinate) {
    std::vector<MeshCoordinate> transfer_coords;
    MeshCoordinate send_coord = device_coordinate;
    MeshCoordinate receive_coord = device_coordinate;
    if (is_sender_device) {
        if (is_leftmost) {  // left
            send_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            receive_coord = MeshCoordinate(device_coordinate.coords()[0] + 1, device_coordinate.coords()[1]);
        } else {  // right
            send_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            receive_coord = MeshCoordinate(device_coordinate.coords()[0] - 1, device_coordinate.coords()[1]);
        }
    } else if (is_root_device) {
        receive_coord = device_coordinate;
        auto sender_coord_1 = MeshCoordinate(device_coordinate.coords()[0] - 1, device_coordinate.coords()[1]);
        auto sender_coord_2 = MeshCoordinate(device_coordinate.coords()[0] + 1, device_coordinate.coords()[1]);
        if (dir == 0) {
            send_coord = sender_coord_2;
        } else {
            send_coord = sender_coord_1;
        }
    } else if (is_root2_device) {
        receive_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
        send_coord = MeshCoordinate(device_coordinate.coords()[0] + 1, device_coordinate.coords()[1]);
        if (dir) {
            send_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            receive_coord = MeshCoordinate(device_coordinate.coords()[0] - 1, device_coordinate.coords()[1]);
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
    const bool is_forward,
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(is_forward);             // is_forward direction
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
    const MeshCoordinate& device_coordinate,
    ReduceToRootOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    auto mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor_l.device());
    // const auto& topology = operation_attributes.topology;
    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_s = tensor_args.input_tensor_s;
    const auto& input_tensor_m = tensor_args.input_tensor_m;
    const auto& intermediate_tensor_l = output_tensors.at(0)[0];
    // const auto& intermediate_tensor_sm = output_tensors.at(1)[1];
    const auto& output_tensor_l = output_tensors.at(1)[0];
    const auto& output_tensor_s = output_tensors.at(1)[1];
    const auto& output_tensor_m = output_tensors.at(1)[2];

    printf("page size of intermediate tensor l: %zu\n", intermediate_tensor_l.tensor_spec().compute_page_size_bytes());

    printf("output tensors len: %zu\n", output_tensors.size());
    printf("len of output tensor [0]: %zu\n", output_tensors.at(0).size());
    printf("len of output tensor [1]: %zu\n", output_tensors.at(1).size());
    printf("print all shapes of output tensors:\n");
    printf(
        "intermediate tensor l shape: %u %u\n",
        output_tensors.at(0)[0].logical_shape()[0],
        output_tensors.at(0)[0].logical_shape()[1]);
    printf(
        "intermediate tensor sm shape: %u %u %u\n",
        output_tensors.at(0)[1].logical_shape()[0],
        output_tensors.at(0)[1].logical_shape()[1],
        output_tensors.at(0)[1].logical_shape()[2]);
    printf(
        "output tensor l shape: %u %u\n",
        output_tensors.at(1)[0].logical_shape()[0],
        output_tensors.at(1)[0].logical_shape()[1]);
    printf(
        "output tensor s shape: %u %u\n",
        output_tensors.at(1)[1].logical_shape()[0],
        output_tensors.at(1)[1].logical_shape()[1]);
    printf(
        "output tensor m shape: %u %u\n",
        output_tensors.at(1)[2].logical_shape()[0],
        output_tensors.at(1)[2].logical_shape()[1]);

    // TODO: fix cb synchronization for receivers to get a whole chunk for tensor l

    // check which device within the column:
    // root device
    // root2 device: the intermediate device (has forward and backward neighbors) but not root
    // senders are leaf devices (only have one neighbor)

    // check which device is this one based on coordinates
    auto device = input_tensor_l.device();
    auto mesh_shape = mesh_device->shape();
    bool is_root_device = false;
    bool is_root2_device = false;
    bool is_sender_device = false;

    if (device_coordinate == root_coord) {
        // this is the root device
        is_root_device = true;
    } else if (
        device_coordinate.coords()[1] == root_coord.coords()[1] &&
        (device_coordinate.coords()[0] == root_coord.coords()[0] - 1 ||
         device_coordinate.coords()[0] == root_coord.coords()[0] + 1) &&
        device_coordinate.coords()[0] != 0 && device_coordinate.coords()[0] != mesh_shape[0] - 1) {
        // this is the intermediate device
        is_root2_device = true;
    } else {
        // this is a sender device
        is_sender_device = true;
    }

    auto semaphore_round1 = semaphores[0];
    auto semaphore_round2 = semaphores[1];
    // when writing program factory make sure:
    //  all devices have 3 input buffers for the inputs
    //  devices 1 and 2 have three intermediate buffers for intermediate results
    //  device 1 has three buffers for the output results
    //  all devices except device 1 have buffers for packet headers and packets

    uint32_t num_shard_cores = 4;  // 8 for 2 links and get the value from the tensor
    uint32_t input_l_total_num_pages = data_movement::get_num_pages(input_tensor_l);
    uint32_t input_l_num_pages = input_l_total_num_pages / num_shard_cores;
    printf("input l total num pages: %u, per core num pages: %u\n", input_l_total_num_pages, input_l_num_pages);

    const uint32_t input_page_size_bytes =
        input_tensor_l.tensor_spec().compute_page_size_bytes();  // same page size: assuming all are tiny tiles
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    printf("input page size bytes: %u\n", input_page_size_bytes);

    // figure out packets
    const auto [packet_size_bytes_initial, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(
            input_tensor_l.dtype(), input_page_size_bytes, input_l_num_pages, l1_alignment);
    printf("packet size bytes initial: %u \n", packet_size_bytes_initial);

    // HERE TODO TO DO MAKE SURE THIS IS CORRECT FOR BLACKHOLE CHANGE IT TO 8192
    uint32_t packet_size_bytes = 2048;  // 8192 = (8 * 512 * 2)
    // eventually add more cores for multi-link
    // const CoreCoord use_cores = {1, 1};
    // const auto
    //    [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2]
    //    =
    //        tt::tt_metal::split_work_to_cores(use_cores, total_packets);
    const auto all_coord_cores = {CoreCoord(0, 0), CoreCoord(0, 1), CoreCoord(0, 2), CoreCoord(0, 3)};
    const CoreRangeSet all_cores = CoreRangeSet(all_coord_cores);
    const uint32_t num_cores = 4;
    // TO DO HERE change above to 8 cores for 2 link

    printf("num cores: %u \n", num_cores);
    printf("total packets: %u \n", total_packets);
    // program!
    tt::tt_metal::Program program{};

    constexpr uint32_t input_num_tiles = 4;  // 2  // to be modified with tiny tiles HERE

    // TODO allocate buffers only on needed devices

    // sdpa compute values
    const auto tile_width = input_tensor_l.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor_l.tensor_spec().tile().get_height();
    printf("tile width: %u, tile height: %u\n", tile_width, tile_height);
    /*
    uint32_t head_dim_v = 64;
    bool use_mla = true;
    // q_shape = {1, 1, 8, 64};
    // k_shape = {1, 8, 64, 64};
    uint32_t PNH = 8, DH = 64;
    uint32_t DHt = DH / tile_width;
    uint32_t vDHt = input_num_tiles;  // use_mla ? head_dim_v / tile_width : DHt;
    printf("vDHt should be: %u\n", use_mla ? head_dim_v / tile_width : DHt);
    uint32_t Sq_chunk_t = PNH / q_heads_parallel_factor / tile_height;
    */
    // questions?
    //  1- use mla?
    //  2- head_dim_v?
    //  3- q_heads_parallel_factor?
    //  4- double check q_shape and k_shape values
    //  5- double check head dim value

    bool use_mla = true;
    uint32_t q_heads_parallel_factor = 1;
    uint32_t head_dim_v = 64;
    // auto q_shape = {1, 1, 8, 64} ; //{1, B, PNH, DH};  //PNH being number of heads = 8
    // auto k_shape = {1, 8, 256, 64}; //{B, NKV, S, DH};  //NKV being number of experts. also assuming S = 256
    uint32_t B = 1;    // q_shape[1];
    uint32_t PNH = 8;  // q_shape[2],
    uint32_t S = 256;  // k_shape[2],
    uint32_t DH = 64;  // k_shape[3];
    printf("B: %u, PNH: %u, S: %u, DH: %u\n", B, PNH, S, DH);
    uint32_t Bkv = 1;  // k_shape[0];
    uint32_t St = S / tile_height;
    uint32_t DHt = DH / tile_width;
    uint32_t vDHt = use_mla ? head_dim_v / tile_width : DHt;
    uint32_t PNHt = PNH / q_heads_parallel_factor / tile_height;

    const uint32_t Sq_chunk_t = 1;  // PNHt;

    printf("Bkv: %u, St: %u, DHt: %u, vDHt: %u, PNHt: %u, Sq_chunk_t: %u\n", Bkv, St, DHt, vDHt, PNHt, Sq_chunk_t);

    uint32_t statistics_tiles = PNHt;
    tt::DataFormat stats_df = tt::DataFormat::Float16_b;
    // const auto full_tile = tt::tt_metal::Tile({32, 32});
    const auto tiny_tile = tt::tt_metal::Tile({8, 32});  // HERE
    auto stats_tile = tiny_tile;                         // full_tile;
    uint32_t stats_tile_size = stats_tile.get_tile_size(stats_df);

    // Create buffers
    constexpr auto sender_cb_l = tt::CBIndex::c_0;
    constexpr auto cb_num_pages = 2;
    constexpr uint32_t chunk_size = input_num_tiles;
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_l.dtype());

    printf("aligned_input_page_size_bytes: %u\n", aligned_input_page_size_bytes);
    printf("statistics_tiles: %u\n", statistics_tiles);
    printf("stats_tile_size: %u\n", stats_tile_size);

    tt::tt_metal::CircularBufferConfig cb_sender_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * chunk_size * aligned_input_page_size_bytes, {{sender_cb_l, input_dataformat}})
            .set_page_size(sender_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(sender_cb_l, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_sender_l_config);

    constexpr auto sender_cb_s = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_sender_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{sender_cb_s, input_dataformat}})
            .set_page_size(sender_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(sender_cb_s, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_sender_s_config);

    constexpr auto sender_cb_m = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_sender_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{sender_cb_m, input_dataformat}})
            .set_page_size(sender_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(sender_cb_m, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_sender_m_config);

    constexpr auto compute_cb_l = tt::CBIndex::c_3;
    constexpr auto cb_compute_num_pages = 2 * input_num_tiles;
    tt::tt_metal::CircularBufferConfig cb_compute_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{compute_cb_l, input_dataformat}})
            .set_page_size(compute_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_l, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_l_config);

    constexpr auto compute_cb_s = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_compute_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_s, input_dataformat}})
            .set_page_size(compute_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_s, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_s_config);

    constexpr auto compute_cb_m = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_compute_m_config =
        tt::tt_metal::CircularBufferConfig(
            statistics_tiles * aligned_input_page_size_bytes, {{compute_cb_m, input_dataformat}})
            .set_page_size(compute_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_m, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_m_config);

    constexpr auto compute_cb_2_l = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig cb_compute_2_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{compute_cb_2_l, input_dataformat}})
            .set_page_size(compute_cb_2_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_l, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_2_l_config);

    constexpr auto compute_cb_2_s = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_compute_2_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_s, input_dataformat}})
            .set_page_size(compute_cb_2_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_s, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_2_s_config);

    constexpr auto compute_cb_2_m = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig cb_compute_2_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_m, input_dataformat}})
            .set_page_size(compute_cb_2_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_cb_2_m, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_2_m_config);

    // allocate space for packet headers for payload sempahore
    constexpr auto packet_header_cb_id = tt::CBIndex::c_9;
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_header_config);

    constexpr auto packet_cb_id = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes)
            .set_tile_dims(packet_cb_id, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_packet_config);

    // intermediate buffers for root and root2
    constexpr auto intermediate_cb_l = tt::CBIndex::c_11;
    const uint32_t intermediate_cb_l_size_bytes = input_num_tiles * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_l_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_l_size_bytes, {{intermediate_cb_l, input_dataformat}})
            .set_page_size(intermediate_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_l, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_intermediate_l_config);

    constexpr auto intermediate_cb_s = tt::CBIndex::c_12;
    const uint32_t intermediate_cb_s_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_s_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_s_size_bytes, {{intermediate_cb_s, input_dataformat}})
            .set_page_size(intermediate_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_s, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_intermediate_s_config);

    constexpr auto intermediate_cb_m = tt::CBIndex::c_13;
    const uint32_t intermediate_cb_m_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_m_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_m_size_bytes, {{intermediate_cb_m, input_dataformat}})
            .set_page_size(intermediate_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(intermediate_cb_m, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_intermediate_m_config);

    constexpr auto compute_out_cb_l = tt::CBIndex::c_14;
    tt::tt_metal::CircularBufferConfig cb_compute_out_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{compute_out_cb_l, input_dataformat}})
            .set_page_size(compute_out_cb_l, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_l, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_out_l_config);

    constexpr auto compute_out_cb_s = tt::CBIndex::c_15;
    tt::tt_metal::CircularBufferConfig cb_compute_out_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_s, input_dataformat}})
            .set_page_size(compute_out_cb_s, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_s, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_out_s_config);

    constexpr auto compute_out_cb_m = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_compute_out_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_m, input_dataformat}})
            .set_page_size(compute_out_cb_m, aligned_input_page_size_bytes)
            .set_tile_dims(compute_out_cb_m, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_compute_out_m_config);

    constexpr auto cb_exp_max_diff_2 = tt::CBIndex::c_17;
    constexpr auto cb_exp_num_pages = 2;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_2_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff_2, input_dataformat}})
            .set_page_size(cb_exp_max_diff_2, aligned_input_page_size_bytes)
            .set_tile_dims(cb_exp_max_diff_2, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_exp_max_diff_2_config);

    constexpr auto cb_exp_max_diff = tt::CBIndex::c_18;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff, input_dataformat}})
            .set_page_size(cb_exp_max_diff, aligned_input_page_size_bytes)
            .set_tile_dims(cb_exp_max_diff, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_exp_max_diff_config);

    constexpr auto cb_m_temp = tt::CBIndex::c_19;
    tt::tt_metal::CircularBufferConfig cb_m_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_m_temp, input_dataformat}})
            .set_page_size(cb_m_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_m_temp, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_m_temp_config);

    // allocate buffers for packet headers and packets for root2
    //  allocate space for packet headers for payload sempahore
    constexpr auto packet_header_cb_id_2 = tt::CBIndex::c_20;
    tt::tt_metal::CircularBufferConfig cb_header_config_2 =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id_2, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id_2, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id_2, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_header_config_2);

    constexpr auto packet_cb_id_2 = tt::CBIndex::c_21;
    tt::tt_metal::CircularBufferConfig cb_packet_config_2 =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id_2, input_dataformat}})
            .set_page_size(packet_cb_id_2, packet_size_bytes)
            .set_tile_dims(packet_cb_id_2, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_packet_config_2);

    constexpr auto cb_s_temp = tt::CBIndex::c_22;
    tt::tt_metal::CircularBufferConfig cb_s_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s_temp, input_dataformat}})
            .set_page_size(cb_s_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s_temp, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_s_temp_config);

    constexpr auto cb_s1_temp = tt::CBIndex::c_23;
    tt::tt_metal::CircularBufferConfig cb_s1_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s1_temp, input_dataformat}})
            .set_page_size(cb_s1_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s1_temp, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_s1_temp_config);

    constexpr auto cb_s2_temp = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig cb_s2_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_s2_temp, input_dataformat}})
            .set_page_size(cb_s2_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_s2_temp, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_s2_temp_config);

    constexpr auto cb_l1_temp = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig cb_l1_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{cb_l1_temp, input_dataformat}})
            .set_page_size(cb_l1_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_l1_temp, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_l1_temp_config);

    constexpr auto cb_l2_temp = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig cb_l2_temp_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{cb_l2_temp, input_dataformat}})
            .set_page_size(cb_l2_temp, aligned_input_page_size_bytes)
            .set_tile_dims(cb_l2_temp, stats_tile);
    // CreateCircularBuffer(program, all_cores, cb_l2_temp_config);

    // create cbs only on needed devices
    if (is_sender_device) {
        CreateCircularBuffer(program, all_cores, cb_sender_l_config);
        CreateCircularBuffer(program, all_cores, cb_sender_s_config);
        CreateCircularBuffer(program, all_cores, cb_sender_m_config);
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

    printf("after creating circular buffers\n");
    // Create different kernels
    tt::tt_metal::KernelHandle reader_kernel = 0;
    tt::tt_metal::KernelHandle writer_kernel = 0;
    std::vector<uint32_t> reader_ct_args;
    std::vector<uint32_t> writer_ct_args;
    std::vector<uint32_t> compute_ct_args;

    // 1. Setup muxes for each device type
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const uint32_t num_workers_per_direction = 1;  // change it to 4 when adding all cores;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    const std::vector<CoreCoord> mux_cores = {CoreCoord(2, 0), CoreCoord(2, 1)};  // to be modified based on device type
    // TODO here change above to 4 cores for 2 links

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
        printf("is sender device satrt\n");
        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/sender_reader_kernel.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

        writer_ct_args = {0, sender_cb_l, sender_cb_s, sender_cb_m, packet_header_cb_id, packet_cb_id, l1_alignment};
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
        printf("is sender device end\n");
    }

    else if (is_root_device) {
        printf("is root device satrt\n");
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
            compute_cb_2_m};
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

        writer_ct_args = {
            compute_out_cb_l,
            compute_out_cb_s,
            compute_out_cb_m,
        };
        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_writer_kernel.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
        printf("is root device end\n");
    } else if (is_root2_device) {
        printf("is root2 device satrt\n");
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
            compute_cb_2_m};
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
            compute_out_cb_l, compute_out_cb_s, compute_out_cb_m, packet_header_cb_id_2, packet_cb_id_2, l1_alignment};
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
        printf("is root2 device end\n");
    }

    if (!is_sender_device) {
        printf("compute kernel start\n");
        // scale = 1/sqrt(head_size)
        uint32_t head_size = 64;
        auto scale_fp32 = static_cast<uint32_t>(1.0f / sqrtf(static_cast<float>(head_size)));
        // TODO: not sure of this value cause it will end up being 0, so I will set to 1 for now
        scale_fp32 = 1;
        uint32_t loop_size = is_root_device ? 2 : 1;
        compute_ct_args = {
            compute_out_cb_l, compute_cb_2_l,   compute_cb_l,      compute_cb_2_s, compute_cb_m,
            compute_cb_2_m,   compute_out_cb_m, cb_exp_max_diff_2, compute_cb_s,   cb_exp_max_diff,
            compute_out_cb_s, cb_m_temp,        cb_s_temp,         cb_s1_temp,     cb_s2_temp,
            cb_l1_temp,       cb_l2_temp,       scale_fp32,        Sq_chunk_t,     vDHt,
            loop_size,
        };
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/compute_kernel2.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .compile_args = compute_ct_args,
            });
        printf("compute kernel end\n");
    }

    printf("before setting runtime args\n");
    // set runtime args
    bool is_leftmost = true;
    if (is_sender_device && device_coordinate.coords()[0] != 0) {
        is_leftmost = false;
    }

    constexpr auto num_links = 1;  // for now TODO change to 2

    // uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> sender_cores;
    std::vector<CoreRangeSet> cores_per_link;
    std::vector<CoreCoord> cores_link_1 = {CoreCoord(0, 0), CoreCoord(0, 1), CoreCoord(0, 2), CoreCoord(0, 3)};
    std::vector<CoreCoord> cores_link_2 = {CoreCoord(1, 0), CoreCoord(1, 1), CoreCoord(1, 2), CoreCoord(1, 3)};
    cores_per_link.push_back(CoreRangeSet(cores_link_1));
    cores_per_link.push_back(CoreRangeSet(cores_link_2));
    CoreCoord termination_master = CoreCoord(0, 0);

    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord virtual_core = link_idx == 0 ? mesh_device->worker_core_from_logical_core(CoreCoord(0, 1))
                                               : mesh_device->worker_core_from_logical_core(CoreCoord(1, 1));
        CoreCoord opposite_core_coord = link_idx == 0 ? mesh_device->worker_core_from_logical_core(CoreCoord(0, 2))
                                                      : mesh_device->worker_core_from_logical_core(CoreCoord(1, 2));
        uint32_t start_ix = link_idx == 0 ? 0 : 2;
        if (link_idx == 1) {
            termination_master = CoreCoord(1, 0);
        }
        for (uint32_t dir = 0; dir < 2; dir++) {
            CoreCoord mux_logical_core = dir == 0 ? CoreCoord(2, start_ix) : CoreCoord(2, start_ix + 1);
            if (mux_connection_valid(dir, is_leftmost, is_sender_device)) {
                std::vector<uint32_t> mux_rt_args = {};
                auto transfer_coords = find_send_recv(
                    dir, is_leftmost, is_sender_device, is_root_device, is_root2_device, device_coordinate);
                auto send_coord = transfer_coords[0];
                auto receive_coord = transfer_coords[1];
                printf("send coord here is: %u %u \n", send_coord.coords()[0], send_coord.coords()[1]);
                printf("receive coord here is: %u %u \n", receive_coord.coords()[0], receive_coord.coords()[1]);
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
                    input_l_num_pages,
                    input_tensor_s.buffer()->address(),
                    input_tensor_m.buffer()->address(),
                    input_page_size_bytes,
                    core_noc_x,
                    core_noc_y};
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);

                writer_runtime_args = {
                    intermediate_tensor_l.buffer()->address(),
                    // intermediate_tensor_sm.buffer()->address(),
                    0,                  // page_idx_start,
                    input_l_num_pages,  // page_idx_end,
                    input_page_size_bytes,
                    packet_size_bytes,
                    num_page_segments,
                    semaphore_round1.address(),
                    core_noc_x,
                    core_noc_y,
                    is_leftmost ? virtual_core.x : opposite_core_coord.x,
                    is_leftmost ? virtual_core.y : opposite_core_coord.y};
                // if leftmost: device 0:
                // only backward direction is valid so start_idx + 1
                // if rightmost: device 3
                // only forward direction is valid so start_idx
                CoreCoord mux_virtual_core =
                    mesh_device->worker_core_from_logical_core(CoreCoord(2, start_ix + !is_leftmost));

                fabric_mux_rt_args(
                    is_leftmost,
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
                printf("is root device setting rt args\n");

                printf("before setting reader rt args\n");

                CoreCoord mux_virtual_core_fwd = mesh_device->worker_core_from_logical_core(CoreCoord(2, start_ix));
                CoreCoord mux_virtual_core_bwd = mesh_device->worker_core_from_logical_core(CoreCoord(2, start_ix + 1));

                reader_runtime_args = {
                    0,  // fabric_2_idx,
                    input_tensor_l.buffer()->address(),
                    input_tensor_s.buffer()->address(),
                    input_tensor_m.buffer()->address(),
                    // intermediate buffers
                    intermediate_cb_l,
                    intermediate_cb_s,
                    intermediate_cb_m,
                    0,                  // page_idx_start,
                    input_l_num_pages,  // page_idx_end,
                    num_pages_per_packet,
                    intermediate_tensor_l.buffer()->address(),
                    // intermediate_tensor_sm.buffer()->address(),
                    packet_size_bytes,
                    input_page_size_bytes,
                    num_page_segments,
                    semaphore_round1.address(),
                    semaphore_round2.address(),
                    1,  // num_hops,
                    core_noc_x,
                    core_noc_y,
                    opposite_core_coord.x,
                    opposite_core_coord.y,
                    virtual_core.x,
                    virtual_core.y,
                };
                printf("before adding fabric rt args\n");

                // first receiving from device on the left
                fabric_mux_rt_args(
                    0,  // first bwd
                    c == termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_bwd,
                    worker_id,
                    c,
                    mux_kernel_config,
                    program,
                    mesh_device->worker_core_from_logical_core(termination_master),
                    reader_runtime_args);

                reader_runtime_args[0] = reader_runtime_args.size();
                printf("reader_rt_args size for fabric  2: %zu\n", reader_ct_args.size());

                // then receiving from device on the right
                fabric_mux_rt_args(
                    1,  // then forward
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
                printf("before setting writer rt args\n");
                printf("intermediate_cb_l: %u\n", intermediate_cb_l);
                printf("input_l_num_pages: %u\n", input_l_num_pages);
                printf("intermediate_cb_s: %u\n", intermediate_cb_s);
                printf("intermediate_cb_m: %u\n", intermediate_cb_m);
                printf("output_tensor_l addr: %u\n", output_tensor_l.buffer()->address());
                printf("output_tensor_s addr: %u\n", output_tensor_s.buffer()->address());
                printf("output_tensor_m addr: %u\n", output_tensor_m.buffer()->address());
                printf("input_page_size_bytes: %u\n", input_page_size_bytes);
                writer_runtime_args = {
                    intermediate_cb_l,
                    input_l_num_pages,
                    intermediate_cb_s,
                    intermediate_cb_m,
                    output_tensor_l.buffer()->address(),
                    output_tensor_s.buffer()->address(),
                    output_tensor_m.buffer()->address(),
                    input_page_size_bytes,
                    core_noc_x,
                    core_noc_y};
                printf("is root device end of setting rt args\n");
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel, c, writer_runtime_args);
            } else if (is_root2_device) {
                CoreCoord mux_virtual_core_fwd = mesh_device->worker_core_from_logical_core(CoreCoord(2, start_ix));
                CoreCoord mux_virtual_core_bwd = mesh_device->worker_core_from_logical_core(CoreCoord(2, start_ix + 1));

                reader_runtime_args = {
                    input_tensor_l.buffer()->address(),
                    input_tensor_s.buffer()->address(),
                    input_tensor_m.buffer()->address(),
                    0,  // page_idx_start,
                    input_l_num_pages,
                    num_pages_per_packet,
                    intermediate_tensor_l.buffer()->address(),
                    // intermediate_tensor_sm.buffer()->address(),
                    packet_size_bytes,
                    input_page_size_bytes,
                    num_page_segments,
                    semaphore_round1.address(),
                    1,  // num_hops,
                    core_noc_x,
                    core_noc_y,
                    virtual_core.x,
                    virtual_core.y,
                };

                fabric_mux_rt_args(
                    1,  // first forward
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
                    intermediate_tensor_l.buffer()->address(),
                    // intermediate_tensor_sm.buffer()->address(),
                    0,                  // page_idx_start,
                    input_l_num_pages,  // page_idx_end,
                    input_page_size_bytes,
                    packet_size_bytes,
                    num_page_segments,
                    semaphore_round2.address(),
                    core_noc_x,
                    core_noc_y,
                    opposite_core_coord.x,
                    opposite_core_coord.y};
                fabric_mux_rt_args(
                    0,  // then backward
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
            // page_idx_start += increment;
            sender_cores.push_back(c);
            printf("after setting runtime args for core (%zu, %zu)\n", c.x, c.y);
        }  // end of loop
        worker_id++;
    }

    return {
        std::move(program),
        ReduceToRootOp::ReduceToRoot::shared_variables_t{//.send_unary_reader_kernel_id = send_unary_reader_kernel_id,
                                                         //.send_unary_writer_kernel_id = send_unary_writer_kernel_id,
                                                         //.sender_cores = sender_cores,
                                                         .semaphores = semaphores}};
}

void ReduceToRootOp::ReduceToRoot::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto send_coord = operation_attributes.root_coord;

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());
        // const auto& shared_variables = cached_workload.shared_variables.at(range);
        /*
        if (coord == send_coord) {
            const auto& send_unary_reader_kernel_id = shared_variables.send_unary_reader_kernel_id;
            const auto& send_unary_writer_kernel_id = shared_variables.send_unary_writer_kernel_id;

            // change this when we use more cores for multi-link
            const auto& core = shared_variables.sender_cores.at(0);

            auto& reader_runtime_args = GetRuntimeArgs(program, send_unary_reader_kernel_id, core);
            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, send_unary_writer_kernel_id, core);
            writer_runtime_args.at(0) = tensor_return_value.at(0).buffer()->address();
            writer_runtime_args.at(8) = shared_variables.semaphore.address();
        }

        if (coord == receive_coord) {
            const auto& receive_unary_reader_kernel_id = shared_variables.receive_unary_reader_kernel_id;
            const auto& receive_unary_writer_kernel_id = shared_variables.receive_unary_writer_kernel_id;

            // change this when we use more cores for multi-link
            const auto& core = shared_variables.receiver_cores.at(0);

            auto& reader_runtime_args = GetRuntimeArgs(program, receive_unary_reader_kernel_id, core);
            reader_runtime_args.at(3) = tensor_return_value.at(0).buffer()->address();
            reader_runtime_args.at(7) = shared_variables.semaphore.address();

            auto& writer_runtime_args = GetRuntimeArgs(program, receive_unary_writer_kernel_id, core);
            writer_runtime_args.at(0) = tensor_return_value.at(1).buffer()->address();
        }
        */
    }
};
}  // namespace ttnn::operations::ccl

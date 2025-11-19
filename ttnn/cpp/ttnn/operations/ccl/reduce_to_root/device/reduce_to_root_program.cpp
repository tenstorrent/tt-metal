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

namespace ttnn::operations::ccl {

ttnn::device_operation::CachedProgram<ReduceToRootOp::ReduceToRoot::shared_variables_t> reduce_to_root_program_factory(
    const ReduceToRootOp::tensor_args_t& tensor_args,
    const ReduceToRootOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& device_coordinate,
    ReduceToRootOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    auto mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor[0].device());
    const auto& topology = operation_attributes.topology;
    const auto& input_tensor_l = tensor_args.input_tensor[0];
    const auto& input_tensor_s = tensor_args.input_tensor[1];
    const auto& input_tensor_m = tensor_args.input_tensor[2];
    const auto& intermediate_tensor_l = output_tensors.at(1)[0];
    const auto& intermediate_tensor_sm = output_tensors.at(1)[1];
    const auto& output_tensor_l = output_tensors.at(0)[0];
    const auto& output_tensor_s = output_tensors.at(0)[1];
    const auto& output_tensor_m = output_tensors.at(0)[2];

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

    // basic accounting
    const uint32_t input_l_num_pages = data_movement::get_num_pages(input_tensor_l);

    const uint32_t input_page_size_bytes =
        input_tensor_l.tensor_spec().compute_page_size_bytes();  // same page size: assuming all are tiny tiles
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // figure out packets
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        detail::compute_aligned_packet_dims(
            input_tensor_l.dtype(), input_page_size_bytes, input_l_num_pages, l1_alignment);

    // eventually add more cores for multi-link
    const CoreCoord use_cores = {1, 1};
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_packets_per_core_group_1, num_packets_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(use_cores, total_packets);

    printf("num cores: %u \n", num_cores);
    printf("total packets: %u \n", total_packets);
    // program!
    tt::tt_metal::Program program{};

    // TODO allocate buffers only on needed devices

    // sdpa compute values
    uint32_t q_heads_parallel_factor = 1;
    const auto tile_width = input_tensor_l.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor_l.tensor_spec().tile().get_height();
    uint32_t head_dim_v = 64;
    bool use_mla = true;
    // q_shape = {1, 1, 8, 64};
    // k_shape = {1, 8, 64, 64};
    uint32_t PNH = 8, DH = 64;
    uint32_t DHt = DH / tile_width;
    uint32_t vDHT = use_mla ? head_dim_v / tile_width : DHt;
    uint32_t Sq_chunk_t = PNH / q_heads_parallel_factor / tile_height;

    // Create buffers
    constexpr auto sender_cb_l = tt::CBIndex::c_0;
    constexpr auto cb_num_pages = 2;
    constexpr uint32_t chunk_size = 8;
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_l.dtype());
    tt::tt_metal::CircularBufferConfig cb_sender_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * chunk_size * aligned_input_page_size_bytes, {{sender_cb_l, input_dataformat}})
            .set_page_size(sender_cb_l, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_sender_l_config);

    constexpr auto sender_cb_s = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_sender_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{sender_cb_s, input_dataformat}})
            .set_page_size(sender_cb_s, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_sender_s_config);

    constexpr auto sender_cb_m = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_sender_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{sender_cb_m, input_dataformat}})
            .set_page_size(sender_cb_m, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_sender_m_config);

    constexpr auto compute_cb_l = tt::CBIndex::c_3;
    constexpr auto cb_compute_num_pages = 8;
    tt::tt_metal::CircularBufferConfig cb_compute_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{compute_cb_l, input_dataformat}})
            .set_page_size(compute_cb_l, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_l_config);

    constexpr auto compute_cb_s = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_compute_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_s, input_dataformat}})
            .set_page_size(compute_cb_s, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_s_config);

    constexpr auto compute_cb_m = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_compute_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_m, input_dataformat}})
            .set_page_size(compute_cb_m, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_m_config);

    constexpr auto compute_cb_2_l = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig cb_compute_2_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{compute_cb_2_l, input_dataformat}})
            .set_page_size(compute_cb_2_l, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_2_l_config);

    constexpr auto compute_cb_2_s = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_compute_2_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_s, input_dataformat}})
            .set_page_size(compute_cb_2_s, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_2_s_config);

    constexpr auto compute_cb_2_m = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig cb_compute_2_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_cb_2_m, input_dataformat}})
            .set_page_size(compute_cb_2_m, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_2_m_config);

    // allocate space for packet headers for payload sempahore
    constexpr auto packet_header_cb_id = tt::CBIndex::c_9;
    constexpr auto buffering_factor = 2;  // this is in other fabric kernels
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_header_config);

    constexpr auto packet_cb_id = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
            .set_page_size(packet_cb_id, packet_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_packet_config);

    // intermediate buffers for root and root2
    constexpr auto intermediate_cb_l = tt::CBIndex::c_11;
    const uint32_t intermediate_cb_l_size_bytes = 8 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_l_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_l_size_bytes, {{intermediate_cb_l, input_dataformat}})
            .set_page_size(intermediate_cb_l, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_intermediate_l_config);

    constexpr auto intermediate_cb_s = tt::CBIndex::c_12;
    const uint32_t intermediate_cb_s_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_s_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_s_size_bytes, {{intermediate_cb_s, input_dataformat}})
            .set_page_size(intermediate_cb_s, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_intermediate_s_config);

    constexpr auto intermediate_cb_m = tt::CBIndex::c_13;
    const uint32_t intermediate_cb_m_size_bytes = 1 * aligned_input_page_size_bytes;
    tt::tt_metal::CircularBufferConfig cb_intermediate_m_config =
        tt::tt_metal::CircularBufferConfig(intermediate_cb_m_size_bytes, {{intermediate_cb_m, input_dataformat}})
            .set_page_size(intermediate_cb_m, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_intermediate_m_config);

    constexpr auto compute_out_cb_l = tt::CBIndex::c_14;
    tt::tt_metal::CircularBufferConfig cb_compute_out_l_config =
        tt::tt_metal::CircularBufferConfig(
            cb_compute_num_pages * aligned_input_page_size_bytes, {{compute_out_cb_l, input_dataformat}})
            .set_page_size(compute_out_cb_l, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_out_l_config);

    constexpr auto compute_out_cb_s = tt::CBIndex::c_15;
    tt::tt_metal::CircularBufferConfig cb_compute_out_s_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_s, input_dataformat}})
            .set_page_size(compute_out_cb_s, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_out_s_config);

    constexpr auto compute_out_cb_m = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_compute_out_m_config =
        tt::tt_metal::CircularBufferConfig(1 * aligned_input_page_size_bytes, {{compute_out_cb_m, input_dataformat}})
            .set_page_size(compute_out_cb_m, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_compute_out_m_config);

    constexpr auto cb_exp_max_diff_2 = tt::CBIndex::c_17;
    constexpr auto cb_exp_num_pages = 8;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_2_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff_2, input_dataformat}})
            .set_page_size(cb_exp_max_diff_2, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_exp_max_diff_2_config);

    constexpr auto cb_exp_max_diff = tt::CBIndex::c_18;
    tt::tt_metal::CircularBufferConfig cb_exp_max_diff_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * aligned_input_page_size_bytes, {{cb_exp_max_diff, input_dataformat}})
            .set_page_size(cb_exp_max_diff, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_exp_max_diff_config);

    constexpr auto cb_out_accumulate_im = tt::CBIndex::c_19;
    tt::tt_metal::CircularBufferConfig cb_out_accumulate_im_config =
        tt::tt_metal::CircularBufferConfig(
            cb_exp_num_pages * vDHT * aligned_input_page_size_bytes, {{cb_out_accumulate_im, input_dataformat}})
            .set_page_size(cb_out_accumulate_im, aligned_input_page_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_accumulate_im_config);

    auto data_core = CoreCoord(0, 0);  // depends where the data is located
    auto data_core_coord = device->worker_core_from_logical_core(data_core);
    auto core_noc_x = data_core_coord.x;
    auto core_noc_y = data_core_coord.y;
    // Create different kernels
    tt::tt_metal::KernelHandle reader_kernel = 0;
    tt::tt_metal::KernelHandle writer_kernel = 0;
    std::vector<uint32_t> reader_ct_args;
    std::vector<uint32_t> writer_ct_args;
    std::vector<uint32_t> compute_ct_args;
    if (is_sender_device) {
        reader_ct_args = {data_core_coord.x, data_core_coord.y};
        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/sender_reader_kernel.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

        writer_ct_args = {
            0,
            sender_cb_l,
            sender_cb_s,
            sender_cb_m,
            packet_header_cb_id,
            packet_cb_id,
            l1_alignment,
            core_noc_x,
            core_noc_y};
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_l.buffer()).append_to(writer_ct_args);
        writer_ct_args[0] = writer_ct_args.size();
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_sm.buffer()).append_to(writer_ct_args);

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
            compute_cb_l,
            compute_cb_s,
            compute_cb_m,
            l1_alignment,
            compute_cb_2_l,
            compute_cb_2_s,
            compute_cb_2_m,
            core_noc_x,
            core_noc_y};
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_l.buffer()).append_to(reader_ct_args);
        reader_ct_args[0] = reader_ct_args.size();
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_sm.buffer()).append_to(reader_ct_args);

        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_reader_kernel.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

        writer_ct_args = {
            core_noc_x,
            core_noc_y,
            compute_out_cb_l,
            compute_out_cb_s,
            compute_out_cb_m,
        };
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
            core_noc_x,
            core_noc_y};
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_l.buffer()).append_to(reader_ct_args);
        reader_ct_args[0] = reader_ct_args.size();
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_sm.buffer()).append_to(reader_ct_args);
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
            packet_header_cb_id,
            packet_cb_id,
            l1_alignment,
            core_noc_x,
            core_noc_y};
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_l.buffer()).append_to(writer_ct_args);
        writer_ct_args[0] = writer_ct_args.size();
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor_sm.buffer()).append_to(writer_ct_args);
        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root2_writer_kernel.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    }

    if (!is_sender_device) {
        // scale = 1/sqrt(head_size)
        uint32_t head_size = 64;
        auto scale_fp32 = static_cast<uint32_t>(1.0f / sqrtf(static_cast<float>(head_size)));
        // TODO: not sure of this value cause it will end up being 0, so I will set to 1 for now
        scale_fp32 = 1;
        uint32_t loop_size = is_root_device ? 2 : 1;
        compute_ct_args = {
            compute_cb_2_l,
            compute_out_cb_l,
            compute_cb_l,
            compute_cb_2_s,
            compute_cb_m,
            compute_cb_2_m,
            compute_out_cb_m,
            cb_exp_max_diff_2,
            compute_cb_s,
            cb_exp_max_diff,
            compute_out_cb_s,
            cb_out_accumulate_im,
            scale_fp32,
            Sq_chunk_t,
            vDHT,
            loop_size,
        };
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/compute_kernel.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .compile_args = compute_ct_args,
            });
    }

    // set runtime args
    MeshCoordinate send_coord = {0, 0};
    MeshCoordinate receive_coord = {0, 0};
    if (is_sender_device) {
        if (device_coordinate.coords()[0] == 0) {
            // left sender
            send_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            receive_coord = MeshCoordinate(device_coordinate.coords()[0] + 1, device_coordinate.coords()[1]);
        } else {
            // right sender
            send_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            receive_coord = MeshCoordinate(device_coordinate.coords()[0] - 1, device_coordinate.coords()[1]);
        }
    }
    constexpr auto link_idx = 0;  // for single link implementation

    // uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> sender_cores;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        /*
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_packets_per_core_group_1 * num_pages_per_packet;
        } else if (core_group_2.contains(c)) {
            increment = num_packets_per_core_group_2 * num_pages_per_packet;
        } else {
            continue;
        }
        increment = std::min(increment, input_num_pages - page_idx_start);
        page_idx_end += increment;
        */

        std::vector<uint32_t> reader_runtime_args;
        std::vector<uint32_t> writer_runtime_args;
        if (is_sender_device) {
            const auto this_fabric_id = mesh_device->get_fabric_node_id(send_coord);

            const auto [num_hops, dst_is_forward, next_fabric_id] =
                detail::fabric_routing(mesh_device, send_coord, receive_coord, topology);

            reader_runtime_args = {
                input_tensor_l.buffer()->address(),
                input_l_num_pages,
                input_tensor_s.buffer()->address(),
                input_tensor_m.buffer()->address(),
                input_page_size_bytes};
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);

            writer_runtime_args = {
                intermediate_tensor_l.buffer()->address(),
                intermediate_tensor_sm.buffer()->address(),
                0,                  // page_idx_start,
                input_l_num_pages,  // page_idx_end,
                input_page_size_bytes,
                packet_size_bytes,
                num_page_segments,
                semaphore_round1.address(),
                dst_is_forward,
            };

            if (dst_is_forward) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id, link_idx, program, c, writer_runtime_args);
            }
            writer_runtime_args.emplace_back(!dst_is_forward);
            if (!dst_is_forward) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id, link_idx, program, c, writer_runtime_args);
            }

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, c, writer_runtime_args);

        } else if (is_root_device) {
            auto receive_coord = root_coord;
            auto sender_coord_1 = MeshCoordinate(root_coord.coords()[0] - 1, root_coord.coords()[1]);
            auto sender_coord_2 = MeshCoordinate(root_coord.coords()[0] + 1, root_coord.coords()[1]);
            const auto this_fabric_id = mesh_device->get_fabric_node_id(receive_coord);

            const auto [num_hops_1, dst_is_forward_1, next_fabric_id_1] =
                detail::fabric_routing(mesh_device, receive_coord, sender_coord_1, topology);

            const auto [num_hops_2, dst_is_forward_2, next_fabric_id_2] =
                detail::fabric_routing(mesh_device, receive_coord, sender_coord_2, topology);

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
                intermediate_tensor_sm.buffer()->address(),
                packet_size_bytes,
                input_page_size_bytes,
                num_page_segments,
                semaphore_round1.address(),
                semaphore_round2.address(),
                1,  // num_hops,
                dst_is_forward_1,
            };

            if (dst_is_forward_1) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id_1, link_idx, program, c, reader_runtime_args);
            }
            reader_runtime_args.emplace_back(!dst_is_forward_1);
            if (!dst_is_forward_1) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id_1, link_idx, program, c, reader_runtime_args);
            }
            reader_runtime_args[0] = reader_runtime_args.size();
            reader_runtime_args.emplace_back(dst_is_forward_2);
            if (dst_is_forward_2) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id_2, link_idx, program, c, reader_runtime_args);
            }
            reader_runtime_args.emplace_back(!dst_is_forward_2);
            if (!dst_is_forward_2) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id_2, link_idx, program, c, reader_runtime_args);
            }

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);

            writer_runtime_args = {
                intermediate_cb_l,
                input_l_num_pages,
                intermediate_cb_s,
                intermediate_cb_m,
                output_tensor_l.buffer()->address(),
                output_tensor_s.buffer()->address(),
                output_tensor_m.buffer()->address(),
                input_page_size_bytes,
            };
        } else if (is_root2_device) {
            auto receive_coord = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            auto sender_coord = MeshCoordinate(device_coordinate.coords()[0] + 1, device_coordinate.coords()[1]);
            const auto this_fabric_id = mesh_device->get_fabric_node_id(receive_coord);

            const auto [num_hops, dst_is_forward, next_fabric_id] =
                detail::fabric_routing(mesh_device, receive_coord, sender_coord, topology);

            reader_runtime_args = {
                input_tensor_l.buffer()->address(),
                input_tensor_s.buffer()->address(),
                input_tensor_m.buffer()->address(),
                0,  // page_idx_start,
                input_l_num_pages,
                num_pages_per_packet,
                intermediate_tensor_l.buffer()->address(),
                intermediate_tensor_sm.buffer()->address(),
                packet_size_bytes,
                input_page_size_bytes,
                num_page_segments,
                semaphore_round1.address(),
                1,  // num_hops,
                dst_is_forward,
            };
            if (dst_is_forward) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id, link_idx, program, c, reader_runtime_args);
            }
            reader_runtime_args.emplace_back(!dst_is_forward);
            if (!dst_is_forward) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id, next_fabric_id, link_idx, program, c, reader_runtime_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, c, reader_runtime_args);

            auto receive_coord_writer = root_coord;
            auto sender_coord_writer = MeshCoordinate(device_coordinate.coords()[0], device_coordinate.coords()[1]);
            const auto this_fabric_id_writer = mesh_device->get_fabric_node_id(sender_coord_writer);

            const auto [num_hops_writer, dst_is_forward_writer, next_fabric_id_writer] =
                detail::fabric_routing(mesh_device, sender_coord_writer, receive_coord_writer, topology);

            writer_runtime_args = {
                intermediate_tensor_l.buffer()->address(),
                intermediate_tensor_sm.buffer()->address(),
                0,                  // page_idx_start,
                input_l_num_pages,  // page_idx_end,
                input_page_size_bytes,
                packet_size_bytes,
                num_page_segments,
                semaphore_round2.address(),
                dst_is_forward_writer,
            };
            if (dst_is_forward_writer) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id_writer, next_fabric_id_writer, link_idx, program, c, writer_runtime_args);
            }
            writer_runtime_args.emplace_back(!dst_is_forward_writer);
            if (!dst_is_forward_writer) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    this_fabric_id_writer, next_fabric_id_writer, link_idx, program, c, writer_runtime_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, c, writer_runtime_args);
        }
        // page_idx_start += increment;
        sender_cores.push_back(c);
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

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

void fabric_mux_connection_ct_args(
    const bool is_termination_master,
    const CoreCoord& mux_virtual_core,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    uint32_t worker_id,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& writer_ct_args) {
    writer_ct_args.push_back(is_termination_master);
    writer_ct_args.push_back(mux_virtual_core.x);
    writer_ct_args.push_back(mux_virtual_core.y);
    writer_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_size_bytes(channel_type));
    writer_ct_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_status_address());
    writer_ct_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_termination_signal_address());
}

void fabric_mux_connection_rt_args(
    const bool& mux_connection_valid,
    const CoreCoord& worker_logical_core,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(mux_connection_valid);
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(termination_master_virtual_core.x);
    worker_rt_args.push_back(termination_master_virtual_core.y);
    worker_rt_args.push_back(num_workers_per_direction);
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    tt::tt_metal::Program program{};
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> empty_fused_op_signaler;
    return all_gather_async_minimal_interleaved_helper(
        program,
        input_tensor,
        sender_device,
        forward_device,
        backward_device,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        sub_device_id,
        empty_fused_op_signaler,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    // Tensor Info
    const auto input_tensor_layout = input_tensor.buffer()->buffer_layout();
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_page_layout = input_tensor.layout();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_layout = output_tensor.buffer()->buffer_layout();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto output_tensor_page_layout = output_tensor.layout();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();

    // op hyperparams
    uint32_t num_workers_per_direction = num_workers_per_direction_opt.value_or(1);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    auto mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    bool fuse_op = fused_op_signaler.has_value();

    // Need a seperate signaler for the sender workers, to handle the first tensor slice that is locally available
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;
    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    TT_FATAL(
        !((topology == ccl::Topology::Linear) && fuse_op), "linear is not support when using fused for all-gather");
    if (topology == ccl::Topology::Ring && ring_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // Get worker cores
    // 2 senders (reader + writer) per direction (forward, backward) per link
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;

    uint32_t num_cores_per_link =
        num_directions_per_link * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
    uint32_t num_workers_per_link = num_directions_per_link * num_workers_per_direction;

    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);
    std::set<CoreRange> sender_worker_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (dir) {
                mux_forward_core_ranges.insert(CoreRange(mux_core));
            } else {
                mux_backward_core_ranges.insert(CoreRange(mux_core));
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                if (dir) {
                    sender_forward_core_ranges.insert(CoreRange(worker_core));
                } else {
                    sender_backward_core_ranges.insert(CoreRange(worker_core));
                }
                sender_worker_core_ranges.insert(CoreRange(worker_core));
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);
    CoreRangeSet mux_forward_core_range_set = CoreRangeSet(mux_forward_core_ranges);
    CoreRangeSet mux_backward_core_range_set = CoreRangeSet(mux_backward_core_ranges);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();

    // scatter-write currently only supports 2 distinct noc addresses, and is only supported for wormhole
    uint32_t max_target_noc_addresses_per_packet = 1;
    if (tt::tt_metal::hal::get_arch() == tt::ARCH::WORMHOLE_B0) {
        max_target_noc_addresses_per_packet = 2;
    }

    // for bfloat8_b, tile_num_per_link=6, we would need to send 2 packages, but they can be of size 3 instead of 4
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t cb_num_pages = 3 * num_tiles_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_sender_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reserved_packet_header_config);

    // KERNEL CREATION
    /* All gather fusion */
    if (fuse_op) {
        auto sender_workers_forward = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
        auto sender_workers_backward = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);
        fused_op_signaler_forward->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
        fused_op_signaler_backward->init_all_gather(
            program, mesh_device, sender_backward_core_ranges, sender_workers_backward);
        fused_op_signaler_sender_workers->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
    }

    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    const uint32_t l1_unreserved_base_address =
        sender_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t batch_head_size = input_tensor_shape[0] * input_tensor_shape[1];

        uint32_t single_batch_head_num_pages = input_tensor_num_pages / batch_head_size;
        TT_FATAL(!(input_tensor_shape[3] % TILE_WIDTH), "Input tensor width must be a multiple of TILE_WIDTH");
        TT_FATAL(!(output_tensor_shape[3] % TILE_WIDTH), "Output tensor width must be a multiple of TILE_WIDTH");
        uint32_t TILE_WIDTH = 32;
        uint32_t input_tensor_Wt = input_tensor_shape[3] / TILE_WIDTH;
        uint32_t input_tensor_Ht = input_tensor_shape[2] / TILE_WIDTH;
        uint32_t output_tensor_Wt = output_tensor_shape[3] / TILE_WIDTH;
        uint32_t output_tensor_Ht = output_tensor_shape[2] / TILE_WIDTH;

        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const bool is_forward = dir;

            // Fabrix mux kernel
            uint32_t mux_core_offset =
                link * num_cores_per_link + dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
            CoreCoord mux_logical_core = all_cores[mux_core_offset];
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
            auto num_full_size_channels = num_workers_per_direction;
            auto num_header_only_channels = 0;
            size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
                num_full_size_channels,
                num_header_only_channels,
                num_buffers_full_size_channels,
                0,
                buffer_size_bytes_full_size_channel,
                mux_base_l1_address);

            const bool mux_connection_valid =
                (dir && backward_device.has_value()) || (!dir && forward_device.has_value());
            if (mux_connection_valid) {
                auto mux_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    {mux_logical_core},
                    tt::tt_metal::DataMovementConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::RISCV_0_default,
                        .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                if (dir) {  // forward
                    const auto dst_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                uint32_t global_worker_id = link * num_workers_per_direction + worker;
                uint32_t global_worker_count = num_links * num_workers_per_direction;
                uint32_t base_pages_per_worker = single_batch_head_num_pages / global_worker_count;
                uint32_t remainder = single_batch_head_num_pages % global_worker_count;
                uint32_t input_tile_id_start =
                    global_worker_id * base_pages_per_worker + std::min(global_worker_id, remainder);
                uint32_t input_tile_id_end =
                    (global_worker_id + 1) * base_pages_per_worker + std::min(global_worker_id + 1, remainder);
                uint32_t chunks_per_sync_val = chunks_per_sync.value_or(
                    std::max((input_tile_id_end - input_tile_id_start) / num_tiles_to_write_per_packet, (uint32_t)1));

                // Reader
                auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
                sender_reader_kernel_config.compile_args = {
                    ring_index,                                        // my_chip_id
                    static_cast<uint32_t>(input_tensor_buffer_type),   // input_buffer_type
                    static_cast<uint32_t>(output_tensor_buffer_type),  // output_buffer_type
                    sender_cb_index,                                   // cb_forward_id
                    num_tiles_to_write_per_packet,                     // num_tiles_to_write_per_packet
                    op_config.get_page_size(),                         // tensor0_page_size
                    num_targets_forward,                               // num_slices_forward_direction
                    num_targets_backward,                              // num_slices_backward_direction
                    static_cast<uint32_t>(topology),                   // topology
                    dir,                                               // direction
                    fuse_op,                                           // fused op
                    chunks_per_sync_val,
                };
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
                    "interleaved_reader.cpp",
                    {core},
                    sender_reader_kernel_config);
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),                         // input_tensor_address
                    output_tensor.buffer()->address(),                        // output_tensor_address
                    input_tensor_Wt,                                          // width in tiles of the output shard
                    input_tensor_Ht,                                          // height in tiles of the output shard
                    output_tensor_Wt,                                         // width in tiles of entire output
                    output_tensor_Ht,                                         // height in tiles of entire output
                    dim,                                                      // dim to gather on
                    batch_head_size,                                          // product of the first two dims
                    input_tile_id_start,                                      //
                    input_tile_id_end,                                        //
                    ring_size,                                                // ring_size
                    semaphore.at(dir).address(),                              // out_ready_semaphore_forward
                    input_tile_id_start % input_tensor_Wt,                    // start_pages_read_in_row
                    input_tile_id_start / input_tensor_Wt * output_tensor_Wt  // start_row_offset
                };
                if (fuse_op) {
                    if (dir) {
                        fused_op_signaler_forward->push_all_gather_fused_op_rt_args(reader_rt_args, 1, 0, 1);
                    } else {
                        fused_op_signaler_backward->push_all_gather_fused_op_rt_args(reader_rt_args, 1, 0, 0);
                    }
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer
                auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
                sender_writer_kernel_config.compile_args = {
                    ring_index,                                        // my_chip_id
                    reserved_packet_header_CB_index,                   // reserved_packet_header_cb_id
                    num_packet_headers_storable,                       // num_packet_headers_storable
                    static_cast<uint32_t>(output_tensor_buffer_type),  // output_buffer_type
                    sender_cb_index,                                   // cb_forward_id
                    num_tiles_to_write_per_packet,                     // num_tiles_to_write_per_packet
                    op_config.get_page_size(),                         // tensor0_page_size
                    num_targets_forward,                               // num_targets_forward_direction
                    num_targets_backward,                              // num_targets_backward_direction
                    dynamic_alternate,                                 // alternate
                    fuse_op,                                           // fused op
                    static_cast<uint32_t>(topology),                   // topology
                    dir,                                               // direction
                    chunks_per_sync_val,
                };
                fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_kernel_config.compile_args);
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
                    "interleaved_writer.cpp",
                    {core},
                    sender_writer_kernel_config);
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

                std::vector<uint32_t> writer_rt_args = {
                    output_tensor.buffer()->address(),                        // output_tensor_address
                    input_tensor_Wt,                                          // width in tiles of the output shard
                    input_tensor_Ht,                                          // height in tiles of the output shard
                    output_tensor_Wt,                                         // width in tiles of entire output
                    output_tensor_Ht,                                         // height in tiles of entire output
                    dim,                                                      // dim to gather on
                    batch_head_size,                                          // product of the first two dims
                    input_tile_id_start,                                      //
                    input_tile_id_end,                                        //
                    virtual_core.x,                                           // out_ready_sem_noc0_x
                    virtual_core.y,                                           // out_ready_sem_noc0_y
                    ring_size,                                                // ring_size
                    semaphore.at(dir).address(),                              // out_ready_semaphore_forward
                    input_tile_id_start % input_tensor_Wt,                    // start_pages_read_in_row
                    input_tile_id_start / input_tensor_Wt * output_tensor_Wt  // start_row_offset
                };
                fabric_mux_connection_rt_args(
                    mux_connection_valid,
                    core,
                    program,
                    termination_master_virtual_core,
                    num_workers_per_direction,
                    writer_rt_args);
                if (fuse_op) {
                    fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_rt_args, 1, 0, 1);
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
            }
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_links,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto out_ready_semaphore_backward = static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore.at(1);
            // update senders
            uint32_t core_idx = 0;
            for (uint32_t link = 0; link < num_links; link++) {
                for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                    for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                        uint32_t mux_core_offset =
                            link * num_cores_per_link +
                            dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
                        CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                        auto out_ready_semaphore =
                            static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore.at(1 - dir);
                        // sender reader
                        auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                        worker_reader_sender_runtime_args[0] = input.buffer()->address();
                        worker_reader_sender_runtime_args[1] = output.buffer()->address();
                        worker_reader_sender_runtime_args[11] = out_ready_semaphore.address();
                        // sender writer
                        auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                        worker_writer_sender_runtime_args[0] = output.buffer()->address();
                        worker_writer_sender_runtime_args[12] = out_ready_semaphore.address();

                        core_idx++;
                    }
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_llama_sharded(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool use_optimal_ccl_for_llama = false) {
    tt::tt_metal::Program program{};

    IDevice* mesh_device = input_tensor.mesh_device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }

    const bool enable_async_output_tensor = false;

    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        use_optimal_ccl_for_llama ? llama_specific::get_custom_worker_core_placement(num_links * num_workers_per_link)
                                  : choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        input_tensor_num_pages / num_links +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // KERNEL CREATION
    // Reader
    auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reader_kernel_config.compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for (const auto& arg : reader_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_reader.cpp",
        sender_worker_core_range,
        reader_kernel_config);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        dynamic_alternate                 // dynamic_alternate
    };
    log_trace(tt::LogOp, "Writer Compile Args:");
    for (const auto& arg : writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);
    auto cores_per_device = output_cores_vec.size() + ring_size - 1 / ring_size;
    uint32_t start_core_index_for_device = output_cores_vec.size() / ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;
    TT_FATAL(
        output_cores_vec.size() % ring_size == 0 || output_cores_vec.size() == 1,
        "output sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution scheme",
        output_cores_vec.size(),
        ring_size);
    auto output_cores_this_device = std::vector<CoreCoord>(
        output_cores_vec.begin() + start_core_index_for_device, output_cores_vec.begin() + end_core_index_for_device);
    log_trace(tt::LogOp, "output_cores_this_device: {}", output_cores_this_device);
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);

        uint32_t worker_num_tiles_to_read = input_tile_id_end - input_tile_id_start;
        uint32_t input_first_core_tile_start_offset = input_tile_id_start % input_tensor_shard_num_pages;
        uint32_t output_first_core_tile_start_offset =
            (input_tensor_num_pages * ring_index + input_tile_id_start) % output_tensor_shard_num_pages;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_tile_id_start / input_tensor_shard_num_pages;
             i < (input_tile_id_end + input_tensor_shard_num_pages - 1) / input_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = input_tile_id_start / output_tensor_shard_num_pages;
             i < (input_tile_id_end + output_tensor_shard_num_pages - 1) / output_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_this_device[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        log_debug(tt::LogOp, "output_first_core_tile_start_offset: {}", output_first_core_tile_start_offset);
        log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        log_debug(tt::LogOp, "output_tensor_cores_x: {}", output_tensor_cores_x);
        log_debug(tt::LogOp, "output_tensor_cores_y: {}", output_tensor_cores_y);

        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }
        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),    // tensor_address0
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;
        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),    // tensor_address0
            semaphore.address(),                  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            const auto src_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
            const auto dst_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            const auto src_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
            const auto dst_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id, worker_sender_writer_kernel_id, semaphore, sender_worker_cores, ring_index](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore.at(0);

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = output.buffer()->address();
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn

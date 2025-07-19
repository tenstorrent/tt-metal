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
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
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
using namespace tt::tt_metal;

namespace ttnn {

using namespace ccl;

void append_fabric_mux_connection_ct_args(
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

void append_fabric_mux_connection_rt_args(
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

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    const std::optional<uint32_t> chunks_per_sync,
    const std::optional<uint32_t> num_workers_per_link,
    const std::optional<uint32_t> num_buffers_per_channel) {
    tt::tt_metal::Program program{};
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler> empty_fused_op_signaler;

    return reduce_scatter_minimal_async_helper(
        program,
        input_tensor,
        intermediate_tensor,
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

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    if (topology == ccl::Topology::Ring) {
        return ring_reduce_scatter_minimal_async_helper(
            program,
            input_tensor,
            intermediate_tensor,
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
            fused_op_signaler,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel);
    } else {
        TT_FATAL(topology == ccl::Topology::Linear, "Must be line or ring");
        return line_reduce_scatter_minimal_async_helper(
            program,
            input_tensor,
            intermediate_tensor,
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
            fused_op_signaler,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            core_grid_offset);
    }
}

tt::tt_metal::operation::ProgramWithCallbacks ring_reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
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

    bool fuse_op = fused_op_signaler.has_value();

    // op hyperparams
    uint32_t num_workers_per_direction = num_workers_per_direction_opt.value_or(1);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {intermediate_tensor, output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores
    // 2 senders per direction (2: forward, backward) per link (num_links)
    // Each sender is reader + compute + writer
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

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto intermediate_tensor_buffer_type = intermediate_tensor.buffer()->buffer_type();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto num_batches = input_tensor_shape[0];
    const auto batch_slice_num_pages = input_tensor_num_pages / ring_size / num_batches;
    const auto batch_slice_num_pages_per_worker = batch_slice_num_pages / (num_workers_per_link * num_links);

    // scatter-write currently only supports 2 distinct noc addresses, and is only supported for wormhole
    uint32_t max_target_noc_addresses_per_packet = 1;
    if (tt::tt_metal::hal::get_arch() == tt::ARCH::WORMHOLE_B0) {
        max_target_noc_addresses_per_packet = 2;
    }

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = num_tiles_to_write_per_packet < 4 ? 4 * num_tiles_to_write_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_input_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_intermediate_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_reader_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_compute_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    static constexpr auto num_packet_headers_storable = 4;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reserved_packet_header_config);

    TT_FATAL(
        !(input_tensor_shape[3] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[3],
        tt::constants::TILE_WIDTH);
    uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;

    // KERNEL CREATION
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    std::vector<KernelHandle> mux_kernel_ids;
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
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

            auto mux_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                {mux_logical_core},
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
            mux_kernel_ids.push_back(mux_kernel_id);

            std::vector<uint32_t> mux_rt_args = {};
            const auto src_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
            if (dir) {  // forward
                const auto dst_node_id =
                    tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            } else {
                const auto dst_node_id =
                    tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            }
            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);

            CoreCoord drain_sync_core;
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                if (worker == 0) {
                    drain_sync_core = virtual_core;
                }

                uint32_t worker_id = link * num_workers_per_direction + worker;
                uint32_t num_workers = num_links * num_workers_per_direction;
                uint32_t tiles_read = (worker_id * batch_slice_num_pages / num_workers);
                uint32_t tiles_to_read = (worker_id + 1) * batch_slice_num_pages / num_workers;
                uint32_t chunks_per_sync_val = chunks_per_sync.value_or(
                    std::max((tiles_to_read - tiles_read) / tile_granularity / 2, (uint32_t)1));

                auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
                sender_reader_kernel_config.compile_args = {
                    ring_index,                                              // my_chip_id
                    static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
                    static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
                    input_cb_index,                                          // cb_input_id
                    intermediate_cb_index,                                   // cb_intermediate_id
                    reader_output_cb_index,                                  // cb_reader_output_id
                    tile_granularity,                                        // packet_size_in_pages
                    op_config.get_page_size(),                               // tensor0_page_size
                    input_tensor_Wt,                                         // input_tensor_Wt
                    batch_slice_num_pages,                                   // batch_slice_num_pages
                    ring_size,                                               // ring_size
                    num_batches,                                             // num_batches
                    fuse_op,                                                 // fused op
                    dir,                                                     // direction
                    chunks_per_sync_val,
                };
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "ring_reduce_scatter_minimal_async_reader.cpp",
                    {core},
                    sender_reader_kernel_config);
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),                 // input_tensor_address
                    intermediate_tensor.buffer()->address(),          // intermediate_tensor_address
                    semaphore.at(dir).address(),                      // out_ready_semaphore
                    semaphore.at(num_directions_per_link).address(),  // batch_ready_semaphore
                    worker_id,
                    num_workers,
                    input_tensor_Wt / ring_size,  // slice_Wt
                    (worker_id * batch_slice_num_pages / num_workers) %
                        (input_tensor_Wt / ring_size),  // start_pages_read_in_row
                    (worker_id * batch_slice_num_pages / num_workers) / (input_tensor_Wt / ring_size) *
                        input_tensor_Wt,                                   // start_row_offset
                    worker_id * batch_slice_num_pages / num_workers,       // start_tiles_read
                    (worker_id + 1) * batch_slice_num_pages / num_workers  // start_tiles_to_read
                };
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer
                auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
                sender_writer_kernel_config.compile_args = {
                    ring_index,                                              // my_chip_id
                    reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
                    num_packet_headers_storable,                             // num_packet_headers_storable
                    static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
                    static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
                    compute_output_cb_index,                                 // cb_compute_output_id
                    reader_output_cb_index,                                  // cb_reader_output_id
                    tile_granularity,                                        // packet_size_in_pages
                    op_config.get_page_size(),                               // tensor0_page_size
                    input_tensor_Wt,                                         // input_tensor_Wt
                    batch_slice_num_pages,                                   // batch_slice_num_pages
                    ring_size,                                               // ring_size
                    num_batches,                                             // num_batches
                    num_tiles_to_write_per_packet,                           // num_tiles_to_write_per_packet
                    dir,                                                     // direction
                    chunks_per_sync_val,
                };
                append_fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_kernel_config.compile_args);
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "ring_reduce_scatter_minimal_async_writer.cpp",
                    {core},
                    sender_writer_kernel_config);
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),          // intermediate_tensor_address
                    output_tensor.buffer()->address(),                // output_tensor_address
                    virtual_core.x,                                   // out_ready_sem_noc0_x
                    virtual_core.y,                                   // out_ready_sem_noc0_y
                    semaphore.at(dir).address(),                      // out_ready_fwd_semaphore
                    semaphore.at(num_directions_per_link).address(),  // batch_ready_semaphore
                    worker_id,
                    num_workers,
                    input_tensor_Wt / ring_size,  // slice_Wt
                    (worker_id * batch_slice_num_pages / num_workers) %
                        (input_tensor_Wt / ring_size),  // pages_read_in_row
                    (worker_id * batch_slice_num_pages / num_workers) / (input_tensor_Wt / ring_size) *
                        input_tensor_Wt,                                   // row_offset
                    (worker_id * batch_slice_num_pages / num_workers),     // tiles_read
                    (worker_id + 1) * batch_slice_num_pages / num_workers  // tiles_to_read
                };
                append_fabric_mux_connection_rt_args(
                    true, core, program, termination_master_virtual_core, num_workers_per_direction, writer_rt_args);
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

                // Reduce kernel
                auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
                sender_reduce_kernel_config.compile_args = {
                    input_cb_index,
                    intermediate_cb_index,
                    compute_output_cb_index,
                    batch_slice_num_pages,
                    tile_granularity,
                    ring_size,
                    num_batches,
                    dir};

                auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "ring_reduction.cpp",
                    {core},
                    sender_reduce_kernel_config);
                reduce_kernel_ids.push_back(sender_reduce_kernel_id);

                std::vector<uint32_t> reduce_rt_args = {
                    worker_id * batch_slice_num_pages / num_workers,       // tiles_read
                    (worker_id + 1) * batch_slice_num_pages / num_workers  // tiles_to_read
                };
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
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
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];

            // update senders
            uint32_t core_idx = 0;
            for (uint32_t link = 0; link < num_links; link++) {
                for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                    for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                        uint32_t mux_core_offset =
                            link * num_cores_per_link +
                            dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
                        CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                        std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                            GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                        std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                            GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                        // sender reader
                        auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                        worker_reader_sender_runtime_args[0] = input.buffer()->address();
                        worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                        // sender writer
                        auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                        worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                        worker_writer_sender_runtime_args[1] = output.buffer()->address();

                        core_idx++;
                    }
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks line_reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    /**
     * Line Reduce Scatter
     *
     *   IN 0     IN 1     IN 2     IN 3            OUT 0    OUT 1    OUT 2    OUT 3
     *   C0       C1       C2       C3              C0       C1       C2       C3
     *  ┌────┐   ┌────┐   ┌────┐   ┌────┐          ┌────┐   ......   ......   ......
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤          └────┘   ┌────┐   ......   ......
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤  ────►   ......   └────┘   ┌────┐   ......
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤          ......   ......   └────┘   ┌────┐
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  └────┘   └────┘   └────┘   └────┘          ......   ......   ......   └────┘
     *
     *
     * There are (ring_size - 1) algorithmic steps in Line Reduce Scatter.
     * Each device must send (num_forward_targets) partials forward and
     * (num_backward_targets) partials backward.
     *
     * On each step, a device will:
     * - if first device in a direction, send a slice in that direction
     * - otherwise, receive a slice, locally reduce it, and send the result in that direction
     *
     */
    auto mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;
    const bool enable_persistent_fabric_mode = true;
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;

    // op hyperparams
    uint32_t num_workers_per_direction = num_workers_per_direction_opt.value_or(1);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    bool fuse_op = fused_op_signaler.has_value();

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {intermediate_tensor, output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores
    // 2 senders (reader + core + writer) per direction (forward, backward) per link
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    // uint32_t num_workers_per_direction = 2;

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
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t max_scatter_write_pages = 1;
    if (tt::tt_metal::hal::get_arch() == tt::ARCH::WORMHOLE_B0) {
        max_scatter_write_pages = 2;
    }
    const uint32_t max_dst_size = 8;  // TODO: generalize based on arch and fp32 acc
    uint32_t tiles_to_write_per_packet = std::min(num_pages_per_packet, max_scatter_write_pages);
    uint32_t tile_granularity = std::min(4 * num_pages_per_packet, max_dst_size);
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_input_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_intermediate_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_reader_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_compute_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    static constexpr auto num_packet_headers_storable = 4;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reserved_packet_header_config);

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto intermediate_tensor_buffer_type = intermediate_tensor.buffer()->buffer_type();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto num_batches = input_tensor_shape[0];
    const auto batch_slice_num_pages = input_tensor_num_pages / ring_size / num_batches;

    TT_FATAL(
        !(input_tensor_shape[3] % tt::constants::TILE_WIDTH),
        "Error, The number of tiles at input tensor dimension {} should be divisible by tile_width but the number of "
        "tiles is {} and the tile_width is {}",
        3,
        input_tensor_shape[3] / tt::constants::TILE_WIDTH,
        tt::constants::TILE_WIDTH);
    uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;

    // KERNEL CREATION
    // Reader
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    std::vector<KernelHandle> mux_kernel_ids;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    uint32_t fwd_bwd_semaphore_address = tt::tt_metal::CreateSemaphore(program, sender_worker_core_range_set, 0);
    const uint32_t l1_unreserved_base_address =
        sender_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    for (uint32_t link = 0; link < num_links; link++) {
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
                (dir && forward_device.has_value()) || (!dir && backward_device.has_value());
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
                mux_kernel_ids.push_back(mux_kernel_id);
                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                if (dir) {  // forward
                    const auto dst_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                // FWD core needs BWD core coordinate for fwd/bwd final reduction sync.
                // For final synchronization, each core needs to know the coordinate of the opposite direction's core.
                uint32_t opposite_mux_core_offset =
                    link * num_cores_per_link +
                    (1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
                uint32_t opposite_core_idx = opposite_mux_core_offset + num_mux_cores_per_direction_per_link + worker;
                auto opposite_core = all_cores[opposite_core_idx];
                auto opposite_core_coord = mesh_device->worker_core_from_logical_core(opposite_core);

                /**
                 * Every chip has a final reduction step. On the ends, there is only one input to reduce.
                 * In the middle, you must reduce from both input directions.
                 * FWD/BWD readers need to synchronize in order to avoid race conditions.
                 *
                 * We'll say that FWD always leads, then signals BWD to follow.
                 */
                const bool is_first_device_in_direction = is_forward ? is_first_chip : is_last_chip;
                const int num_targets_in_direction = is_forward ? num_targets_forward : num_targets_backward;
                // The number of reduction steps is 0 for chips on the ends, or num_targets_in_direction for chips in
                // the middle
                const int num_intermediate_reduction_steps =
                    is_first_device_in_direction ? 0 : num_targets_in_direction;
                const bool do_final_reduction = !is_first_device_in_direction;
                const int num_total_reduction_steps = num_intermediate_reduction_steps + (do_final_reduction ? 1 : 0);
                const bool sync_with_other_direction = !(is_first_chip || is_last_chip);
                uint32_t tiles_read =
                    ((link * num_workers_per_direction + worker) * batch_slice_num_pages /
                     (num_links * num_workers_per_direction));
                uint32_t tiles_to_read = (link * num_workers_per_direction + worker + 1) * batch_slice_num_pages /
                                         (num_links * num_workers_per_direction);
                uint32_t chunks_per_sync_val =
                    chunks_per_sync.value_or(std::max((tiles_to_read - tiles_read) / tile_granularity, (uint32_t)1));

                // Reader
                auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
                sender_reader_kernel_config.compile_args = {
                    ring_index,                                              // my_chip_id
                    static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
                    static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
                    static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
                    input_cb_index,                                          // cb_input_id
                    intermediate_cb_index,                                   // cb_intermediate_id
                    reader_output_cb_index,                                  // cb_reader_output_id
                    tile_granularity,                                        // packet_size_in_pages
                    op_config.get_page_size(),                               // tensor0_page_size
                    input_tensor_Wt,                                         // input_tensor_Wt
                    batch_slice_num_pages,                                   // batch_slice_num_pages
                    ring_size,                                               // ring_size
                    num_batches,                                             // num_batches
                    fuse_op,                                                 // fused op
                    tiles_to_write_per_packet,                               // contig_pages_advanced
                    is_forward,                                              // direction
                    is_first_device_in_direction,
                    num_targets_in_direction,
                    num_intermediate_reduction_steps,
                    do_final_reduction,
                    num_total_reduction_steps,
                    sync_with_other_direction,
                    chunks_per_sync_val,
                };
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "line_reduce_scatter_minimal_async_reader.cpp",
                    {core},
                    sender_reader_kernel_config);
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);
                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    output_tensor.buffer()->address(),        // output_tensor_address
                    semaphore.at(0 + link * 3).address(),     // remote transfer sync semaphore
                    link * num_workers_per_direction + worker,
                    num_links * num_workers_per_direction,
                    fwd_bwd_semaphore_address};
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);
                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);
                // Writer
                auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
                sender_writer_kernel_config.compile_args = {
                    ring_index,                                              // my_chip_id
                    reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
                    num_packet_headers_storable,                             // num_packet_headers_storable
                    static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
                    static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
                    compute_output_cb_index,                                 // cb_compute_output_id
                    reader_output_cb_index,                                  // cb_reader_output_id
                    tile_granularity,                                        // packet_size_in_pages
                    op_config.get_page_size(),                               // tensor0_page_size
                    input_tensor_Wt,                                         // input_tensor_Wt
                    batch_slice_num_pages,                                   // batch_slice_num_pages
                    ring_size,                                               // ring_size
                    num_batches,                                             // num_batches
                    tiles_to_write_per_packet,                               // contig_pages_advanced
                    is_forward,                                              // direction
                    is_first_device_in_direction,
                    num_targets_in_direction,
                    num_intermediate_reduction_steps,
                    do_final_reduction,
                    num_total_reduction_steps,
                    sync_with_other_direction,
                    chunks_per_sync_val,
                };
                append_fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_kernel_config.compile_args);
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "line_reduce_scatter_minimal_async_writer.cpp",
                    {core},
                    sender_writer_kernel_config);
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);
                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    output_tensor.buffer()->address(),        // output_tensor_address
                    virtual_core.x,                           // out_ready_sem_noc0_x
                    virtual_core.y,                           // out_ready_sem_noc0_y
                    semaphore.at(0 + link * 3).address(),     // remote transfer sync semaphore
                    semaphore.at(1 + link * 3).address(),     // final reduction slot semaphore
                    semaphore.at(2 + link * 3).address(),     // batch_ready_semaphore
                    link * num_workers_per_direction + worker,
                    num_links * num_workers_per_direction,
                    fwd_bwd_semaphore_address,
                    opposite_core_coord.x,
                    opposite_core_coord.y};
                append_fabric_mux_connection_rt_args(
                    mux_connection_valid,
                    core,
                    program,
                    termination_master_virtual_core,
                    num_workers_per_direction,
                    writer_rt_args);
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
                // Reduce kernel
                auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
                sender_reduce_kernel_config.compile_args = {
                    input_cb_index,
                    intermediate_cb_index,
                    compute_output_cb_index,
                    batch_slice_num_pages,
                    tile_granularity,
                    ring_size,
                    num_batches,
                    num_links * num_workers_per_direction,
                    num_total_reduction_steps};
                auto reduce_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "line_reduction.cpp",
                    {core},
                    sender_reduce_kernel_config);
                reduce_kernel_ids.push_back(reduce_kernel_id);

                std::vector<uint32_t> reduce_rt_args = {link * num_workers_per_direction + worker};
                tt::tt_metal::SetRuntimeArgs(program, reduce_kernel_id, {core}, reduce_rt_args);
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
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];

            // update senders
            uint32_t core_idx = 0;
            for (uint32_t link = 0; link < num_links; link++) {
                for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                    for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                        uint32_t mux_core_offset =
                            link * num_cores_per_link +
                            dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
                        CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                        std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                            GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                        std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                            GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                        // sender reader
                        auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                        worker_reader_sender_runtime_args[0] = input.buffer()->address();
                        worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                        worker_reader_sender_runtime_args[2] = output.buffer()->address();
                        // sender writer
                        auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                        worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                        worker_writer_sender_runtime_args[1] = output.buffer()->address();

                        core_idx++;
                    }
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace ttnn

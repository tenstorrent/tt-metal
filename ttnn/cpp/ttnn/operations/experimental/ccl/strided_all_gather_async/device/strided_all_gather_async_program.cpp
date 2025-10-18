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
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
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

namespace detail {

uint32_t strided_all_gather_async_core_count_per_link(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    return (num_workers_per_direction + num_mux_cores_per_direction_per_link) * num_directions_per_link;
}

uint32_t strided_default_workers(
    const MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    ccl::Topology topology,
    uint32_t output_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    auto sd_id = sub_device_id.value_or(mesh_device.get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    uint32_t num_cores = subdevice_core_range_set.num_cores();
    // Above 4 workers we start getting performance drops, so we limit to 4 workers or less, depending on the number of
    // available cores This was determined by the sweep
    // tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_T3K.py
    ttnn::SmallVector<uint32_t> candidate_worker_counts;
    // if per link data moved is greater than 0.25 MB, we search greedily for 4 workers, otherwise we search greedily
    // for 2 workers. for ring, half the data is moved per link, so we divide by 2
    double data_moved_per_link_bytes = double(output_data_size_bytes) * (ring_size - 1) / ring_size / num_links /
                                       (topology == ccl::Topology::Ring ? 2 : 1);
    if (data_moved_per_link_bytes > double(0.25 * 1024 * 1024)) {
        candidate_worker_counts = {4, 2, 1};
    } else {
        candidate_worker_counts = {2, 1};
    }
    for (auto worker_count : candidate_worker_counts) {
        uint32_t core_count =
            num_links * strided_all_gather_async_core_count_per_link(
                            worker_count, num_directions_per_link, num_mux_cores_per_direction_per_link);
        if (num_cores >= core_count) {
            log_trace(
                tt::LogOp,
                "data_moved_per_link_bytes: {} and worker_count: {}",
                data_moved_per_link_bytes,
                worker_count);
            return worker_count;
        }
    }
    TT_THROW(
        "Not enough cores available on the subdevice or device for the requested match the number of links {}",
        num_links);
}
}  // namespace detail

using namespace ccl;

void strided_fabric_mux_connection_ct_args(
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

void strided_fabric_mux_connection_rt_args(
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

tt::tt_metal::operation::ProgramWithCallbacks strided_all_gather_async_minimal_default(
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
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    tt::tt_metal::Program program{};
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> empty_fused_op_signaler;
    return strided_all_gather_async_minimal_default_helper(
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
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        empty_fused_op_signaler,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        CoreCoord(0, 0));
}

tt::tt_metal::operation::ProgramWithCallbacks strided_all_gather_async_minimal_default_helper(
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
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    auto mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device not found");

    // op hyperparams
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    // Get worker cores
    // 2 senders (reader + writer) per direction (forward, reverse_order) per link
    uint32_t output_data_size_bytes = output_tensor.buffer()->size();
    uint32_t num_workers_per_direction = num_workers_per_direction_opt.value_or(detail::strided_default_workers(
        *mesh_device,
        sub_device_id,
        topology,
        output_data_size_bytes,
        num_links,
        ring_size,
        num_directions_per_link,
        num_mux_cores_per_direction_per_link));
    uint32_t num_cores_per_link = detail::strided_all_gather_async_core_count_per_link(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    bool fuse_op = fused_op_signaler.has_value();

    // Need a separate signaler for the sender workers, to handle the first tensor slice that is locally available
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;
    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [num_targets_forward, num_targets_backward] =
        ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, false);
    auto [unicast_forward_args, unicast_backward_args] =
        ccl::get_forward_backward_line_unicast_configuration(topology, sender_device, forward_device, backward_device);
    auto [barrier_mcast_forward_args, barrier_mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        topology,
        sender_device,
        forward_device,
        backward_device,
        topology == ccl::Topology::Linear ? num_targets_forward : ring_size - 1,
        topology == ccl::Topology::Linear ? num_targets_backward : ring_size - 1);

    TT_FATAL(
        !((topology == ccl::Topology::Linear) && fuse_op), "linear is not support when using fused for all-gather");

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
    uint32_t l1_scratch_cb_page_size_bytes = page_size;

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

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
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_sender_config);

    bool input_is_sharded = input_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        reader_compute_defines["OUTPUT_IS_SHARDED"] = "1";
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

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
        uint32_t input_tensor_C = input_tensor_shape[1];

        uint32_t output_tensor_Wt = output_tensor_shape[3] / TILE_WIDTH;
        uint32_t output_tensor_Ht = output_tensor_shape[2] / TILE_WIDTH;
        uint32_t output_tensor_C = output_tensor_shape[1];

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
                CoreCoord supplemental_core = all_cores
                    [link * num_cores_per_link +
                     (1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction) +
                     num_mux_cores_per_direction_per_link + worker];
                CoreCoord opposite_core_coord = mesh_device->worker_core_from_logical_core(supplemental_core);

                uint32_t global_worker_id = link * num_workers_per_direction + worker;
                uint32_t global_worker_count = num_links * num_workers_per_direction;
                uint32_t base_pages_per_worker = single_batch_head_num_pages / global_worker_count;
                uint32_t remainder = single_batch_head_num_pages % global_worker_count;
                uint32_t input_tile_id_start =
                    global_worker_id * base_pages_per_worker + std::min(global_worker_id, remainder);
                uint32_t input_tile_id_end =
                    (global_worker_id + 1) * base_pages_per_worker + std::min(global_worker_id + 1, remainder);

                // Heuristic is based on a sweep of large shapes. This will be used when total chunks per worker is
                // larger than 160. Doing it less frequently adds performance cost to many shapes. Sweep test:
                // tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_T3K.py
                constexpr uint32_t HEURISTIC_MAX_CHUNKS_PER_SYNC = 160;
                uint32_t chunks_per_sync_val = chunks_per_sync.value_or(std::min(
                    std::max((input_tile_id_end - input_tile_id_start) / num_tiles_to_write_per_packet, (uint32_t)1),
                    HEURISTIC_MAX_CHUNKS_PER_SYNC));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                uint32_t self_write_done_semaphore;
                if (fuse_op) {
                    self_write_done_semaphore = CreateSemaphore(program, {core}, 0);
                }

                // Reader
                std::vector<uint32_t> sender_reader_compile_args = {
                    ring_index,                       // my_chip_id
                    sender_cb_index,                  // cb_forward_id
                    num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
                    page_size,                        // tensor0_page_size
                    num_targets_forward,              // num_slices_forward_direction
                    num_targets_backward,             // num_slices_backward_direction
                    static_cast<uint32_t>(topology),  // topology
                    dir,                              // direction
                    fuse_op,                          // fused op
                    chunks_per_sync_val,
                    false,
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(output_tensor, sender_reader_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_reader_compile_args);
                }
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/"
                    "minimal_default_reader.cpp",
                    {core},
                    tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),                         // input_tensor_address
                    output_tensor.buffer()->address(),                        // output_tensor_address
                    input_tensor_Wt,                                          // width in tiles of the output shard
                    input_tensor_Ht,                                          // height in tiles of the output shard
                    input_tensor_C,                                           // num input channels
                    output_tensor_Wt,                                         // width in tiles of entire output
                    output_tensor_Ht,                                         // height in tiles of entire output
                    output_tensor_C,                                          // num output channels
                    dim,                                                      // dim to gather on
                    batch_head_size,                                          // product of the first two dims
                    input_tile_id_start,                                      //
                    input_tile_id_end,                                        //
                    ring_size,                                                // ring_size
                    semaphore.at(dir).address(),                              // out_ready_semaphore_forward
                    input_tile_id_start % input_tensor_Wt,                    // start_pages_read_in_row
                    input_tile_id_start / input_tensor_Wt * output_tensor_Wt  // start_row_offset
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    reader_rt_args.push_back(self_write_done_semaphore);
                    if (dir) {
                        fused_op_signaler_forward->push_all_gather_fused_op_rt_args(
                            reader_rt_args,
                            num_workers_per_direction * num_links,
                            worker + link * num_workers_per_direction,
                            1);
                    } else {
                        fused_op_signaler_backward->push_all_gather_fused_op_rt_args(
                            reader_rt_args,
                            num_workers_per_direction * num_links,
                            worker + link * num_workers_per_direction,
                            0);
                    }
                }

                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer
                std::vector<uint32_t> sender_writer_compile_args = {
                    ring_index,                       // my_chip_id
                    sender_cb_index,                  // cb_forward_id
                    num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
                    page_size,                        // tensor0_page_size
                    num_targets_forward,              // num_targets_forward_direction
                    num_targets_backward,             // num_targets_backward_direction
                    fuse_op,                          // fused op
                    static_cast<uint32_t>(topology),  // topology
                    dir,                              // direction
                    chunks_per_sync_val,
                    false,
                };
                strided_fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_compile_args);
                if (dir) {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(),
                        barrier_mcast_backward_args.begin(),
                        barrier_mcast_backward_args.end());
                } else {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(),
                        barrier_mcast_forward_args.begin(),
                        barrier_mcast_forward_args.end());
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
                }
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/"
                    "minimal_default_writer.cpp",
                    {core},
                    tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

                std::vector<uint32_t> writer_rt_args = {
                    output_tensor.buffer()->address(),                           // output_tensor_address
                    input_tensor_Wt,                                             // width in tiles of the output shard
                    input_tensor_Ht,                                             // height in tiles of the output shard
                    input_tensor_C,                                              // num input channels
                    output_tensor_Wt,                                            // width in tiles of entire output
                    output_tensor_Ht,                                            // height in tiles of entire output
                    output_tensor_C,                                             // num output channels
                    dim,                                                         // dim to gather on
                    batch_head_size,                                             // product of the first two dims
                    input_tile_id_start,                                         //
                    input_tile_id_end,                                           //
                    virtual_core.x,                                              // out_ready_sem_noc0_x
                    virtual_core.y,                                              // out_ready_sem_noc0_y
                    ring_size,                                                   // ring_size
                    semaphore.at(dir).address(),                                 // out_ready_semaphore_forward
                    input_tile_id_start % input_tensor_Wt,                       // start_pages_read_in_row
                    input_tile_id_start / input_tensor_Wt * output_tensor_Wt,    // start_row_offset
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use synchronize barrier semaphore
                    barrier_semaphore.has_value()                                // synchronize barrier semaphore
                        ? barrier_semaphore.value().address()
                        : 0,
                    opposite_core_coord.x,
                    opposite_core_coord.y};
                strided_fabric_mux_connection_rt_args(
                    mux_connection_valid,
                    core,
                    program,
                    termination_master_virtual_core,
                    num_workers_per_direction,
                    writer_rt_args);
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }
                if (fuse_op) {
                    writer_rt_args.push_back(self_write_done_semaphore);
                    fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                        writer_rt_args,
                        num_workers_per_direction * num_links,
                        worker + link * num_workers_per_direction,
                        1);
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

            auto barrier_semaphore = static_cast<const ttnn::StridedAllGatherAsync*>(operation)->barrier_semaphore;
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
                            static_cast<const ttnn::StridedAllGatherAsync*>(operation)->semaphore.at(dir);
                        // sender reader
                        auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                        worker_reader_sender_runtime_args[0] = input.buffer()->address();
                        worker_reader_sender_runtime_args[1] = output.buffer()->address();
                        worker_reader_sender_runtime_args[13] = out_ready_semaphore.address();
                        // sender writer
                        auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                        worker_writer_sender_runtime_args[0] = output.buffer()->address();
                        worker_writer_sender_runtime_args[14] = out_ready_semaphore.address();

                        if (barrier_semaphore.has_value()) {
                            worker_writer_sender_runtime_args[18] = barrier_semaphore.value().address();
                        }

                        core_idx++;
                    }
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

static void print_tensor_slice(const ttnn::ccl::v2::TensorSlice& slice_v2) {
    log_trace(tt::LogOp, "TensorSlice:");
    log_trace(
        tt::LogOp,
        "  tensor_shape:        [w={}, z={}, y={}, x={}]",
        slice_v2.tensor_shape.w,
        slice_v2.tensor_shape.z,
        slice_v2.tensor_shape.y,
        slice_v2.tensor_shape.x);
    log_trace(
        tt::LogOp,
        "  tensor_slice_shape:  [w={}, z={}, y={}, x={}]",
        slice_v2.tensor_slice_shape.w,
        slice_v2.tensor_slice_shape.z,
        slice_v2.tensor_slice_shape.y,
        slice_v2.tensor_slice_shape.x);
    log_trace(
        tt::LogOp,
        "  tensor_slice_offset: [w={}, z={}, y={}, x={}]",
        slice_v2.tensor_slice_offset.w,
        slice_v2.tensor_slice_offset.z,
        slice_v2.tensor_slice_offset.y,
        slice_v2.tensor_slice_offset.x);
    log_trace(
        tt::LogOp,
        "  worker_slice_shape:  [w={}, z={}, y={}, x={}]",
        slice_v2.worker_slice_shape.w,
        slice_v2.worker_slice_shape.z,
        slice_v2.worker_slice_shape.y,
        slice_v2.worker_slice_shape.x);
    log_trace(
        tt::LogOp,
        "  worker_slice_offset: [w={}, z={}, y={}, x={}]",
        slice_v2.worker_slice_offset.w,
        slice_v2.worker_slice_offset.z,
        slice_v2.worker_slice_offset.y,
        slice_v2.worker_slice_offset.x);
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord core_grid_offset) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    const size_t num_workers_preferred = num_workers_per_link * num_links;
    const auto available_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    if (available_cores.num_cores() < num_workers_preferred) {
        log_warning(
            tt::LogOp,
            "AllGather is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
            "cores ({} per link and {} links) are made available but only {} are available. This may lead to "
            "performance loss.",
            num_workers_preferred,
            num_workers_per_link,
            num_links,
            available_cores.num_cores());
    }
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                sender_worker_core_range = sender_worker_core_range.merge(CoreRangeSet(CoreRange(
                    CoreCoord(x + core_grid_offset.x, y + core_grid_offset.y),
                    CoreCoord(x + core_grid_offset.x, y + core_grid_offset.y))));
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
            if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                break;
            }
        }
        if (sender_worker_core_range.num_cores() == num_workers_preferred) {
            break;
        }
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_multi_core_with_workers(
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
    const GlobalSemaphore semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};
    IDevice* mesh_device = input_tensor.mesh_device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }
    const bool enable_async_output_tensor = false;
    const bool lower_command_stream_to_noc_commands =
        ttnn::ccl::worker_detail::can_command_stream_be_lowered_to_noc_commands(input_tensor);

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

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create Tensor slicer
    // read the entire input tensor (partition size = 1, partition index = 0)
    // write to the output tensor on its corresponding partition (partition size = ring_size, partition index =
    // ring_index)
    auto input_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicerV2(
        input_tensor,
        dim,
        0,         // partition index
        1,         // partition size
        num_links  // num_workers_per_slicer, set 1 per link for now
    );
    auto output_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicerV2(
        output_tensor,
        dim,
        ring_index,  // partition index
        ring_size,   // partition size
        num_links    // num_workers_per_slicer, set 1 per link for now
    );

    // KERNEL CREATION
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&input_tensor},
            sender_worker_core_range,
            tt::tt_metal::ReaderDataMovementConfig{},
            1,  // num_command_streams
            sender_device->id());

    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&output_tensor},
            sender_worker_core_range,
            tt::tt_metal::WriterDataMovementConfig{},
            1,  // num_command_streams
            sender_device->id());
    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    if (topology == ccl::Topology::Linear) {
        LineTopology line_topology(ring_size, ring_index);
        num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
        num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        if (ring_index % 2 == 0) {
            std::swap(num_targets_forward, num_targets_backward);
        }
    }

    ttnn::ccl::cmd::MulticastCommandDestArgs mcast_dest_args = {num_targets_forward, num_targets_backward};
    log_trace(
        tt::LogOp,
        "[mcast_dest_args] num target forward: {}, num target backward: {}",
        mcast_dest_args.num_targets_forward_direction,
        mcast_dest_args.num_targets_backward_direction);

    auto reader_tensor_slices =
        ttnn::ccl::cmd::builder::generate_worker_tensor_slices(1, input_tensor, num_workers_per_link * num_links, dim);
    log_trace(tt::LogOp, "reader_tensor_slices size: {}", reader_tensor_slices.size());
    log_trace(tt::LogOp, "reader_tensor_slices[0] size: {}", reader_tensor_slices[0].size());

    CoreCoord drain_sync_core;
    // For now these are a little disconnected from the commands - they'll need to be unified and explicitly
    // associated with each other but this is for bootstrapping the feature
    constexpr size_t reader_tensor_command_map_idx = 0;
    constexpr size_t writer_tensor_command_map_idx = 1;
    std::unordered_map<CoreCoord, ttnn::ccl::tensor_address_runtime_args_overrider> reader_rt_args_overrider_map;
    std::unordered_map<CoreCoord, ttnn::ccl::tensor_address_runtime_args_overrider> writer_rt_args_overrider_map;

    for (std::size_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }
        std::size_t worker_tensor_slice_index = link;

        const auto& input_worker_slice_v2 = input_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);
        const auto& output_worker_slice_v2 = output_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);

        log_trace(tt::LogOp, "DEBUG: input tensor slice v2:");
        print_tensor_slice(input_worker_slice_v2);
        log_trace(tt::LogOp, "DEBUG: output tensor slice v2:");
        print_tensor_slice(output_worker_slice_v2);

        log_trace(
            tt::LogOp,
            "DEBUG: ring_index: {}, ring_size: {}, forward_fabric_connection: {}",
            ring_index,
            ring_size,
            forward_device.has_value());
        log_trace(
            tt::LogOp,
            "DEBUG: ring_index: {}, ring_size: {}, backward_fabric_connection: {}",
            ring_index,
            ring_size,
            backward_device.has_value());

        // READER COMMAND STREAM and RT ARGS
        std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> reader_cmd_stream;
        reader_cmd_stream.push_back(  // use the reader_tensor_slices after the bug is fixed
            ttnn::ccl::cmd::uops::read_tensor_slice_to_cb_for_eventual_fabric_write(
                input_worker_slice_v2, src0_cb_index));

        if (lower_command_stream_to_noc_commands) {
            reader_cmd_stream =
                ttnn::ccl::tensor_slice_commands_to_noc_commands(reader_cmd_stream, input_tensor, packet_size_bytes);
        }
        ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
            program,
            worker_sender_reader_kernel_id,
            {&input_tensor},
            {op_config.get_page_size()},
            sender_device,
            link,
            num_pages_per_packet,
            {core},
            reader_cmd_stream,
            std::nullopt,                                        // cmd stream 1
            std::nullopt,                                        // fabric fwd connection
            std::nullopt,                                        // fabric bwd connection
            std::nullopt,                                        // tensor device override
            std::vector<size_t>{reader_tensor_command_map_idx},  // tensor indices
            &reader_rt_args_overrider_map[core]);

        // WRITER COMMAND STREAM and RT ARGS
        std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> writer_cmd_stream;
        // 1, do mcast of the tensor slice to all the destinations
        writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_write_cb_to_tensor_slice(
            output_worker_slice_v2, src0_cb_index, mcast_dest_args));
        // 2, mcast the semaphore to all dest for teardown
        writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_multicast_semaphore_inc(
            &semaphore, ttnn::ccl::cmd::CclCommandAtomicInc{1}, drain_sync_core.x, drain_sync_core.y, mcast_dest_args));
        bool wait_for_semaphore = !enable_async_output_tensor && link == 0;
        if (wait_for_semaphore) {
            // 3, wait for n_chip*num_links number of semaphore at teardown semaphore address for first chip, and
            // n_chip*num_links+1 for other chips
            writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_semaphore_wait(
                &semaphore, is_first_chip ? ring_size * num_links : ring_size * num_links + 1));
            // 4, send semaphore unicast to forward device except for the last chip
            if (!is_last_chip) {
                writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_unicast_semaphore_inc(
                    &semaphore,
                    ttnn::ccl::cmd::CclCommandAtomicInc{1},
                    drain_sync_core.x,
                    drain_sync_core.y,
                    ttnn::ccl::cmd::UnicastCommandDestArgs{1, true}));
            }
        }

        bool reset_semaphore = !enable_async_output_tensor && link == 0;

        if (reset_semaphore) {
            // 6. (drain sync core) reset semaphore to 0
            writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_core_semaphore_set(&semaphore, 0));
        }

        if (lower_command_stream_to_noc_commands) {
            writer_cmd_stream =
                ttnn::ccl::tensor_slice_commands_to_noc_commands(writer_cmd_stream, output_tensor, packet_size_bytes);
        }

        // set the rt args
        ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
            program,
            worker_sender_writer_kernel_id,
            {&output_tensor},
            {op_config.get_page_size()},
            sender_device,
            link,
            num_pages_per_packet,  // num_pages_per_edm_buffer
            {core},
            writer_cmd_stream,
            std::nullopt,
            forward_device,
            backward_device,
            std::nullopt,
            std::vector<size_t>{writer_tensor_command_map_idx},  // tensor indices
            &writer_rt_args_overrider_map[core]);
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         reader_rt_args_overrider_map,
         writer_rt_args_overrider_map,
         reader_tensor_command_map_idx,
         writer_tensor_command_map_idx,
         worker_sender_writer_kernel_id,
         semaphore,
         sender_worker_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                reader_rt_args_overrider_map.at(core).override_runtime_args(
                    reader_tensor_command_map_idx, input.buffer()->address(), worker_reader_sender_runtime_args);
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                writer_rt_args_overrider_map.at(core).override_runtime_args(
                    writer_tensor_command_map_idx, output.buffer()->address(), worker_writer_sender_runtime_args);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn

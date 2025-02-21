// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
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
#include <tt-metalium/mesh_graph.hpp>

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"

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
    bool persistent_fabric_mode,
    IDevice* device,
    const std::optional<SubDeviceId>& sub_device_id) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    if (persistent_fabric_mode) {
        const size_t num_workers_preferred = num_workers_per_link * num_links;
        const auto available_cores = device->worker_cores(
            HalProgrammableCoreType::TENSIX,
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
                    sender_worker_core_range =
                        sender_worker_core_range.merge(CoreRangeSet(CoreRange(CoreCoord(x, y), CoreCoord(x, y))));
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
    } else {
        sender_worker_core_range =
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_workers_per_link - 1, num_links - 1)));
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_async_multi_core_with_workers(
    const Tensor& input_tensor,
    // std::optional<IDevice*> forward_device,
    // std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t mesh_size,
    const mesh_id_t mesh_id,
    ccl::Topology topology,
    const GlobalSemaphore semaphore,
    const std::optional<SubDeviceId>& sub_device_id,  // diff between sub_device_id and device->id() ??
    bool enable_persistent_fabric_mode) {
    tt::tt_metal::Program program{};
    const bool enable_async_output_tensor = false;
    const bool lower_command_stream_to_noc_commands =
        ttnn::ccl::worker_detail::can_command_stream_be_lowered_to_noc_commands(input_tensor);

    IDevice* device = input_tensor.device();

    uint32_t num_chips_per_row = 4;  // should be an input
    uint32_t num_chips_per_col = 4;  // should be an input
    uint32_t direction = 0;          // should be input 0 for horizontally and 1 vertically

    // bool is_first_chip = ring_index == 0;
    // bool is_last_chip = ring_index == ring_size - 1;
    if (direction == 0) {
        bool is_first_chip = mesh_id % num_chips_per_row == 0;
        bool is_last_chip = mesh_id % num_chips_per_row == num_chips_per_row - 1;
        RoutingDirection direction0 = RoutingDirection::E;
        RoutingDirection direction1 = RoutingDirection::W;
    } else {
        bool is_first_chip = mesh_id % num_chips_per_col == 0;
        bool is_last_chip = mesh_id % num_chips_per_col == num_chips_per_col - 1;
        RoutingDirection direction0 = RoutingDirection::N;
        RoutingDirection direction1 = RoutingDirection::S;
    }

    ControlPlane control_plane = DevicePool::instance().get_control_plane();
    std::pair<mesh_id_t, chip_id_t> device_mesh_chip_id =
        control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());  // not sure if sub_device_id instead

    auto neighbors_dir0 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction0);
    auto neighbors_dir1 =
        control_plane->get_intra_chip_neighbors(device_mesh_chip_id.first, device_mesh_chip_id.second, direction1);
    uint32_t hops_dir0 = neighbors_dir0.size();
    uint32_t hops_dir1 = neighbors_dir1.size();
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    mcast_hops[direction0] = hops_dir0;
    mcast_hops[direction1] = hops_dir1;
    IDevice* dir0_device;
    IDevice* dir1_device;
    if (neighbors_dir0.size() > 0) {
        dir0_device = DevicePool::instance().get_device(neighbors_dir0[0]);
    }
    if (neighbors_dir1.size() > 0) {
        dir1_device = DevicePool::instance().get_device(neighbors_dir1[0]);
    }

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            dir0_device.value_or(nullptr),
            dir1_device.value_or(nullptr),
            &program,
            enable_persistent_fabric_mode,
            num_links);

    // LineTopology line_topology(ring_size, ring_index);

    std::unique_ptr<ccl::CclOpTensorConfig> input_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<ccl::CclOpTensorConfig> output_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);

    bool is_sharded = input_tensor.is_sharded();

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, enable_persistent_fabric_mode, device, sub_device_id);

    // L1 Scratch CB Creation
    // packet header seems to have a constant value in tt_fabric_interface.h (not sure)
    const size_t packet_size_bytes = PACKET_HEADER_SIZE_BYTES;  // local_fabric_handle->get_edm_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

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
    uint32_t partition_index;
    uint32_t partition_size;
    if (direction == 0) {
        partition_index = mesh_id % num_chips_per_row;
        partition_size = num_chips_per_row;
    } else {
        partition_index = mesh_id % num_chips_per_col;
        partition_size = num_chips_per_col
    }
    auto output_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicerV2(
        output_tensor,
        dim,
        output_index,    // partition index
        partition_size,  // partition size
        num_links        // num_workers_per_slicer, set 1 per link for now
    );

    // KERNEL CREATION
    KernelHandle worker_sender_reader_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&input_tensor},
            sender_worker_core_range,
            tt::tt_metal::ReaderDataMovementConfig{},
            1,  // num_command_streams
            device->id());

    KernelHandle worker_sender_writer_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&output_tensor},
            sender_worker_core_range,
            tt::tt_metal::WriterDataMovementConfig{},
            1,  // num_command_streams
            device->id());

    // const size_t forward_direction_distance_to_end_of_line =
    //     line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
    // const size_t backward_direction_distance_to_end_of_line =
    //     line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);
    const size_t forward_direction_distance_to_end_of_line = hops_dir0;
    const size_t backward_direction_distance_to_end_of_line = hops_dir1;

    ttnn::ccl::cmd::MulticastCommandDestArgs mcast_dest_args = {
        forward_direction_distance_to_end_of_line, backward_direction_distance_to_end_of_line};
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
            drain_sync_core = device->worker_core_from_logical_core(core);
        }
        std::size_t worker_tensor_slice_index = link;

        const auto& input_worker_slice_v2 = input_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);
        const auto& output_worker_slice_v2 = output_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);

        log_trace(tt::LogOp, "DEBUG: input tensor slice v2:");
        print_tensor_slice(input_worker_slice_v2);
        log_trace(tt::LogOp, "DEBUG: output tensor slice v2:");
        print_tensor_slice(output_worker_slice_v2);

        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> dir0_fabric_connection;  //=some_api_call
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> dir1_fabric_connection;  // some api call
        /*
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> forward_fabric_connection =
            line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                ? std::nullopt
                : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> backward_fabric_connection =
            line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                ? std::nullopt
                : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

        log_trace(
            tt::LogOp,
            "DEBUG: line_index: {}, line_size: {}, forward_fabric_connection: {}",
            line_topology.line_index(),
            line_topology.line_size(),
            forward_fabric_connection.has_value());
        log_trace(
            tt::LogOp,
            "DEBUG: line_index: {}, line_size: {}, backward_fabric_connection: {}",
            line_topology.line_index(),
            line_topology.line_size(),
            backward_fabric_connection.has_value());
        */

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
            input_tensor.device(),
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
        bool generate_teardown_commands = !enable_persistent_fabric_mode && link == 0;
        if (generate_teardown_commands) {
            // 5, increment the termination semaphore for local device for local teardown only for the drain sync core
            auto termination_infos = local_fabric_handle->generate_local_chip_fabric_termination_infos(device);
            for (auto& info : termination_infos) {
                if (info.distance != 0) {
                    continue;
                }
                writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_chip_noc_absolute_address_semaphore_inc(
                    info.edm_noc_x, info.edm_noc_y, info.termination_addr, 1));
            }
        }
        bool reset_semaphore = generate_teardown_commands || (!enable_async_output_tensor && link == 0);
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
            output_tensor.device(),
            num_pages_per_packet,  // num_pages_per_edm_buffer
            {core},
            writer_cmd_stream,
            std::nullopt,
            {dir0_fabric_connection},
            {dir1_fabric_connection},
            std::nullopt,
            std::vector<size_t>{writer_tensor_command_map_idx},  // tensor indices
            &writer_rt_args_overrider_map[core]);
    }

    if (!enable_persistent_fabric_mode) {
        local_fabric_handle->build_kernels();
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

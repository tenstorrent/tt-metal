// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.hpp"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>

#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"

#include <optional>
using namespace tt::constants;

namespace ttnn {

using namespace ccl;

static void print_tensor_slice(ttnn::ccl::v2::TensorSlice const& slice_v2) {
    tt::log_trace(tt::LogOp,"TensorSlice:");
    tt::log_trace(
        tt::LogOp,
        "  tensor_shape:        [w={}, z={}, y={}, x={}]",
        slice_v2.tensor_shape.w,
        slice_v2.tensor_shape.z,
        slice_v2.tensor_shape.y,
        slice_v2.tensor_shape.x);
    tt::log_trace(
        tt::LogOp,
        "  tensor_slice_shape:  [w={}, z={}, y={}, x={}]",
        slice_v2.tensor_slice_shape.w,
        slice_v2.tensor_slice_shape.z,
        slice_v2.tensor_slice_shape.y,
        slice_v2.tensor_slice_shape.x);
    tt::log_trace(
        tt::LogOp,
        "  tensor_slice_offset: [w={}, z={}, y={}, x={}]",
        slice_v2.tensor_slice_offset.w,
        slice_v2.tensor_slice_offset.z,
        slice_v2.tensor_slice_offset.y,
        slice_v2.tensor_slice_offset.x);
    tt::log_trace(
        tt::LogOp,
        "  worker_slice_shape:  [w={}, z={}, y={}, x={}]",
        slice_v2.worker_slice_shape.w,
        slice_v2.worker_slice_shape.z,
        slice_v2.worker_slice_shape.y,
        slice_v2.worker_slice_shape.x);
    tt::log_trace(
        tt::LogOp,
        "  worker_slice_offset: [w={}, z={}, y={}, x={}]",
        slice_v2.worker_slice_offset.w,
        slice_v2.worker_slice_offset.z,
        slice_v2.worker_slice_offset.y,
        slice_v2.worker_slice_offset.x);
}

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_multi_core_with_workers_new(
    const Tensor& input_tensor,
    std::optional<Device*> forward_device,
    std::optional<Device*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    std::optional<GlobalSemaphore> semaphore_handle,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle) {
    bool persistent_fabric_mode = fabric_handle.has_value();
    tt::tt_metal::Program program{};

    // Sleep for ring_index * 5 seconds to stagger startup
    std::this_thread::sleep_for(std::chrono::seconds(ring_index * 5));

    Device* device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    if (!persistent_fabric_mode) {
        fabric_handle =
            ccl::EdmLineFabricOpInterface(device, forward_device, backward_device, &program, false, num_links);
    }

    LineTopology line_topology(ring_size, ring_index);

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
    auto const& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto const& input_tensor_partition = ttnn::ccl::TensorPartition(1, 0);  // one partition, 0 index
    auto const& output_tensor_partition =
        ttnn::ccl::TensorPartition(ring_size, ring_index);  // ring_size partitions, ring_index index

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    auto const& sender_worker_core_range =
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_workers_per_link - 1, num_links - 1)));
    auto const& sender_worker_cores = corerange_to_cores(sender_worker_core_range, std::nullopt, true);

    // L1 Scratch CB Creation
    uint32_t num_pages_per_packet = 1;                 // we assume 1 page per packet for now
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size() + sizeof(tt::fabric::PacketHeader);
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create Tensor slicer
    // read the entire input tensor (partition size = 1, partition index = 0)
    // write to the output tensor on its corresponding partition (partition size = ring_size, partition index =
    // ring_index)
    auto input_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicer(
        input_tensor,
        input_tensor,
        dim,
        0,           // partition index
        1,           // partition size
        num_links,   // num_workers_per_slicer, set 1 per link for now
        UINT32_MAX,  // max_worker_slice_in_bytes, set as infinite for now
        cb_num_pages / 2);
    auto output_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicer(
        output_tensor,
        output_tensor,
        dim,
        ring_index,  // partition index
        ring_size,   // partition size
        num_links,   // num_workers_per_slicer, set 1 per link for now
        UINT32_MAX,  // max_worker_slice_in_bytes, set as infinite for now
        cb_num_pages / 2);

    // KERNEL CREATION
    auto worker_arg_builder = ccl::worker_detail::CCLWorkerArgBuilder(
        device, op_config, input_tensor_partition, output_tensor_partition, dim);

    auto const& worker_defines = op_config.emit_worker_defines();
    static std::string const& sender_kernel_reader_path =
        "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader.cpp";
    static std::string const& sender_kernel_writer_path =
        "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_writer.cpp";

    KernelHandle worker_sender_reader_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&input_tensor},
            sender_worker_core_range,
            tt::tt_metal::ReaderDataMovementConfig{},
            1  // num_command_streams
        );

    KernelHandle worker_sender_writer_kernel_id =
        ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {src0_cb_index},
            {&output_tensor},
            sender_worker_core_range,
            tt::tt_metal::WriterDataMovementConfig{},
            1  // num_command_streams
        );

    const size_t forward_direction_distance_to_end_of_line =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
    const size_t backward_direction_distance_to_end_of_line =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);

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
    for (std::size_t link = 0; link < num_links; link++) {
        CoreCoord core = {num_workers_per_link - 1, link};
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = device->worker_core_from_logical_core(core);
        }
        std::size_t worker_tensor_slice_index = link;
        auto const& input_worker_slice = input_tensor_slicer.get_worker_slice(worker_tensor_slice_index);
        auto const& output_worker_slice = output_tensor_slicer.get_worker_slice(worker_tensor_slice_index);
        auto worker_arg_builder = ccl::worker_detail::CCLWorkerArgBuilder(
            device, op_config, input_tensor_partition, output_tensor_partition, dim);

        // tt::log_trace("Creating RT Args for worker core ({},{})", core.x, core.y);
        log_trace(tt::LogOp, "reference input worker slice");
        input_worker_slice.print();
        log_trace(tt::LogOp, "reference output worker slice");
        output_worker_slice.print();

        auto const& input_worker_slice_v2 = input_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);
        auto const& output_worker_slice_v2 = output_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);

        log_trace(tt::LogOp, "DEBUG: input tensor slice v2:");
        print_tensor_slice(input_worker_slice_v2);
        log_trace(tt::LogOp, "DEBUG: output tensor slice v2:");
        print_tensor_slice(output_worker_slice_v2);

        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> forward_fabric_connection =
            line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                ? std::nullopt
                : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(
                      fabric_handle->uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> backward_fabric_connection =
            line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
                ? std::nullopt
                : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(
                      fabric_handle->uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

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

        // READER COMMAND STREAM and RT ARGS
        std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> reader_cmd_stream;
        reader_cmd_stream.push_back(  // use the reader_tensor_slices after the bug is fixed
            ttnn::ccl::cmd::uops::read_tensor_slice_to_cb_for_eventual_fabric_write(
                input_worker_slice_v2, src0_cb_index));

        ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
            program,
            worker_sender_reader_kernel_id,
            {&input_tensor},
            {op_config.get_page_size()},
            input_tensor.device(),
            num_pages_per_packet,
            {core},
            reader_cmd_stream,
            std::nullopt,
            std::nullopt,
            std::nullopt);

        // WRITER COMMAND STREAM and RT ARGS
        std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> writer_cmd_stream;
        // 1, do mcast of the tensor slice to all the destinations
        writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_write_cb_to_tensor_slice(
            output_worker_slice_v2, src0_cb_index, mcast_dest_args));
        // 2, mcast the semaphore to all dest for teardown
        bool generate_teardown_commands = !persistent_fabric_mode && link == 0;
        if (generate_teardown_commands) {
            TT_FATAL(
                semaphore_handle.has_value(),
                "Internal error during all-=gather fatcory. Global semaphore for fabric teardown not properly "
                "initialized for non-persistent fabric mode");
            writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_multicast_semaphore_inc(
                semaphore_handle.value(),
                ttnn::ccl::cmd::CclCommandAtomicInc{1},
                drain_sync_core.x,
                drain_sync_core.y,
                mcast_dest_args));
            // 3, wait for n_chip*num_links number of semaphore at teardown semaphore address for first chip, and
            // n_chip*num_links+1 for other chips
            writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_semaphore_wait(
                semaphore_handle.value(), is_first_chip ? ring_size * num_links : ring_size * num_links + 1));
            // 4, send semaphore unicast to forward device except for the last chip
            if (!is_last_chip) {
                writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::fabric_unicast_semaphore_inc(
                    semaphore_handle.value(),
                    ttnn::ccl::cmd::CclCommandAtomicInc{1},
                    drain_sync_core.x,
                    drain_sync_core.y,
                    ttnn::ccl::cmd::UnicastCommandDestArgs{1, true}));
            }
            // 5, increment the termination semaphore for local device for local teardown only for the drain sync core
            auto termination_infos = fabric_handle->generate_local_chip_fabric_termination_infos(device);
            for (auto& info : termination_infos) {
                if (info.distance != 0) {
                    continue;
                }
                writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_chip_noc_absolute_address_semaphore_inc(
                    info.edm_noc_x, info.edm_noc_y, info.termination_addr, 1));
            }
            // 6. (drain sync core) reset semaphore to 0
            writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_core_semaphore_set(semaphore_handle.value(), 0));
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
            {forward_fabric_connection},
            {backward_fabric_connection});
    }

    if (!persistent_fabric_mode) {
        fabric_handle->build_kernels();
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id, worker_sender_writer_kernel_id, semaphore_handle, sender_worker_cores](
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
            for (auto const& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args.at(0) = input.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args.at(0) = output.buffer()->address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include <algorithm>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

namespace ttnn::experimental::prim {

RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::cached_program_shared_variable_t
RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> empty_fused_op_signaler;
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto* mesh_device = tensor_args.input_tensor[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : mesh_device;
    std::vector<IDevice*> devices_to_use = {};
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    const auto& mesh_view = mesh_device->get_view();
    devices_to_use = (operation_attributes.cluster_axis.value() == 0)
                         ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                         : mesh_view.get_devices_on_row(mesh_coordinate[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }
    auto
        [worker_sender_reader_forward_kernel_id,
         worker_sender_writer_forward_kernel_id,
         worker_sender_reader_backward_kernel_id,
         worker_sender_writer_backward_kernel_id,
         sender_worker_cores,
         num_inputs,
         reader_sender_rt_offset,
         writer_sender_rt_offset,
         num_links] =
            ring_attention_all_gather_async_multi_core_with_workers_helper(
                program,
                tensor_args.input_tensor,
                target_device,
                forward_device,
                backward_device,
                tensor_return_value,
                operation_attributes.dim,
                operation_attributes.num_links,
                operation_attributes.ring_size,
                device_index,
                operation_attributes.topology,
                operation_attributes.semaphore,
                operation_attributes.sub_device_id,
                empty_fused_op_signaler);

    shared_variables_t shared_variables{
        .worker_sender_reader_forward_kernel_id = worker_sender_reader_forward_kernel_id,
        .worker_sender_writer_forward_kernel_id = worker_sender_writer_forward_kernel_id,
        .worker_sender_reader_backward_kernel_id = worker_sender_reader_backward_kernel_id,
        .worker_sender_writer_backward_kernel_id = worker_sender_writer_backward_kernel_id,
        .sender_worker_cores = sender_worker_cores,
        .num_inputs = num_inputs,
        .reader_sender_rt_offset = reader_sender_rt_offset,
        .writer_sender_rt_offset = writer_sender_rt_offset,
        .num_links = num_links,
    };

    return {std::move(program), std::move(shared_variables)};
}

RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::cached_mesh_workload_t
RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

void RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_program.workload.get_programs()) {
        auto& shared_variables = cached_program.shared_variables.at(coordinate_range);
        const auto& input_tensors = tensor_args.input_tensor;
        const auto& output_tensors = tensor_return_value;
        const auto& semaphore = operation_attributes.semaphore;

        ring_attention_all_gather_async_multicore_with_workers_override_runtime_arguments(
            shared_variables, program, input_tensors, output_tensors, semaphore);
    }
}

}  // namespace ttnn::experimental::prim

namespace ttnn {

RingAttentionAllGatherAsyncMultiCoreWithWorkersSharedVariables
ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::Program& program,
    const std::vector<Tensor>& input_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    std::vector<Tensor>& output_tensor,
    int32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset) {
    auto* mesh_device = input_tensor[0].device();
    [[maybe_unused]] const bool is_first_chip = ring_index == 0;
    [[maybe_unused]] const bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.at(0).device()->id(),
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    const bool fuse_op = fused_op_signaler.has_value();

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;

    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = input_tensor;
    const std::vector<Tensor>& output_tensors = output_tensor;
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    if (topology == ttnn::ccl::Topology::Ring && ring_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // Get worker cores
    // 2 sender (forward/backward, each with a reader/writer)
    uint32_t num_senders_per_link = 2;
    const auto [sender_worker_core_range, sender_worker_cores] =
        ttnn::ccl::choose_worker_cores(num_links, num_senders_per_link, mesh_device, sub_device_id, core_grid_offset);

    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;

    for (int i = 0; i < sender_worker_cores.size(); i++) {
        const auto& core = sender_worker_cores[i];
        if (i % 2 == 1) {
            sender_forward_core_ranges.insert(CoreRange(core));
        } else {
            sender_backward_core_ranges.insert(CoreRange(core));
        }
    }

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    const uint32_t max_scatter_write_pages = 2;
    const uint32_t num_pages_per_packet =
        std::min((uint32_t)(packet_size_bytes / l1_scratch_cb_page_size_bytes), max_scatter_write_pages);
    const uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor[0].dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_forward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_forward_cb_index, df}})
            .set_page_size(sender_forward_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_forward_core_ranges, cb_sender_forward_config);
    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_sender_backward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_backward_cb_index, df}})
            .set_page_size(sender_backward_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_backward_core_ranges, cb_sender_backward_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_forward_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_forward_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_forward_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_forward_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_forward_core_ranges, cb_reserved_packet_header_forward_config);
    const auto reserved_packet_header_backward_CB_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_backward_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_backward_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_backward_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_backward_core_ranges, cb_reserved_packet_header_backward_config);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor[0].buffer()->num_pages();
    const auto input_tensor_shape = input_tensor[0].padded_shape();
    const auto output_tensor_shape = output_tensor[0].padded_shape();
    const uint32_t num_inputs = input_tensor.size();

    uint32_t tiles_to_write_per_packet = 1;
    // KERNEL CREATION
    // Forward Direction
    // Reader
    auto sender_reader_forward_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_reader_forward_kernel_config.compile_args = {
        ring_index,                       // my_chip_id
        sender_forward_cb_index,          // cb_forward_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        tiles_to_write_per_packet,        // contig_pages_advanced
        num_inputs,                       // num_inputs
        1,                                // direction
        fuse_op,                          // fused op
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_reader_forward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel_config.compile_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel_config.compile_args);
    }
    auto worker_sender_reader_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp",
        sender_forward_core_ranges,
        sender_reader_forward_kernel_config);

    // Writer
    auto sender_writer_forward_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_writer_forward_kernel_config.compile_args = {
        ring_index,                               // my_chip_id
        reserved_packet_header_forward_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,              // num_packet_headers_storable
        sender_forward_cb_index,                  // cb_forward_id
        num_pages_per_packet,                     // packet_size_in_pages
        op_config.get_page_size(),                // tensor0_page_size
        num_targets_forward,                      // num_targets_forward_direction
        num_targets_backward,                     // num_targets_backward_direction
        dynamic_alternate,                        // alternate
        fuse_op,                                  // fused op
        static_cast<uint32_t>(topology),          // topology
        tiles_to_write_per_packet,                // contig_pages_advanced
        num_inputs,                               // num_inputs
        1,                                        // direction
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_forward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_forward_kernel_config.compile_args);
    }
    auto worker_sender_writer_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp",
        sender_forward_core_ranges,
        sender_writer_forward_kernel_config);

    // Backward Direction
    // Reader
    auto sender_reader_backward_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_reader_backward_kernel_config.compile_args = {
        ring_index,                       // my_chip_id
        sender_backward_cb_index,         // cb_backward_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        tiles_to_write_per_packet,        // contig_pages_advanced
        num_inputs,                       // num_inputs
        0,                                // direction
        fuse_op,                          // fused op
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_reader_backward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel_config.compile_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel_config.compile_args);
    }
    auto worker_sender_reader_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp",
        sender_backward_core_ranges,
        sender_reader_backward_kernel_config);

    // Writer
    auto sender_writer_backward_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_writer_backward_kernel_config.compile_args = {
        ring_index,                                // my_chip_id
        reserved_packet_header_backward_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,               // num_packet_headers_storable
        sender_backward_cb_index,                  // cb_backward_id
        num_pages_per_packet,                      // packet_size_in_pages
        op_config.get_page_size(),                 // tensor0_page_size
        num_targets_forward,                       // num_targets_forward_direction
        num_targets_backward,                      // num_targets_backward_direction
        dynamic_alternate,                         // alternate
        fuse_op,                                   // fused op
        static_cast<uint32_t>(topology),           // topology
        tiles_to_write_per_packet,                 // contig_pages_advanced
        num_inputs,                                // num_inputs
        0,                                         // direction
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_backward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_backward_kernel_config.compile_args);
    }
    auto worker_sender_writer_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp",
        sender_backward_core_ranges,
        sender_writer_backward_kernel_config);

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
    // Kernel Runtime Args
    uint32_t reader_sender_rt_offset = 0;
    uint32_t writer_sender_rt_offset = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        // Set Sender Reader runtime args
        const uint32_t batch_head_size = input_tensor_shape[0] * input_tensor_shape[1];

        uint32_t single_batch_head_num_pages = input_tensor_num_pages / batch_head_size;
        const uint32_t base_pages_per_worker = single_batch_head_num_pages / num_links;
        const uint32_t remainder = single_batch_head_num_pages % num_links;
        const uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        const uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

        TT_ASSERT(!(input_tensor_shape[3] % tt::constants::TILE_WIDTH));
        TT_ASSERT(!(output_tensor_shape[3] % tt::constants::TILE_WIDTH));
        const uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;
        const uint32_t input_tensor_Ht = input_tensor_shape[2] / tt::constants::TILE_WIDTH;
        const uint32_t output_tensor_Wt = output_tensor_shape[3] / tt::constants::TILE_WIDTH;
        const uint32_t output_tensor_Ht = output_tensor_shape[2] / tt::constants::TILE_WIDTH;

        std::vector<uint32_t> reader_forward_rt_args = {
            input_tensor_Wt,            // width in tiles of the input shard
            input_tensor_Ht,            // height in tiles of the input shard
            output_tensor_Wt,           // width in tiles of the entire output
            output_tensor_Ht,           // height in tiles of the entire output
            dim,                        // dim to gather on
            batch_head_size,            // product of the first two dims
            input_tile_id_start,        //
            input_tile_id_end,          // slice_num_pages
            ring_size,                  // ring_size
            semaphore.at(1).address(),  // out_ready_semaphore_backward
        };
        reader_sender_rt_offset = reader_forward_rt_args.size();
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(input_tensor[input_idx].buffer()->address());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        if (fuse_op) {
            fused_op_signaler_forward->push_all_gather_fused_op_rt_args(reader_forward_rt_args, num_links, link, 1);
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_reader_forward_kernel_id,
            {sender_worker_cores[(link * 2) + 1]},
            reader_forward_rt_args);

        std::vector<uint32_t> reader_backward_rt_args = {
            input_tensor_Wt,            // width in tiles of the input shard
            input_tensor_Ht,            // height in tiles of the input shard
            output_tensor_Wt,           // width in tiles of the entire output
            output_tensor_Ht,           // height in tiles of the entire output
            dim,                        // dim to gather on
            batch_head_size,            // product of the first two dims
            input_tile_id_start,        // slice_num_pages
            input_tile_id_end,          // slice_num_pages
            ring_size,                  // ring_size
            semaphore.at(0).address(),  // out_ready_semaphore_backward
        };
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(input_tensor[input_idx].buffer()->address());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        if (fuse_op) {
            fused_op_signaler_backward->push_all_gather_fused_op_rt_args(reader_backward_rt_args, num_links, link, 0);
        }
        tt::tt_metal::SetRuntimeArgs(
            program, worker_sender_reader_backward_kernel_id, {sender_worker_cores[link * 2]}, reader_backward_rt_args);

        const CoreCoord sender_forward_worker_core =
            mesh_device->worker_core_from_logical_core(sender_worker_cores[(link * 2) + 1]);
        const CoreCoord sender_backward_worker_core =
            mesh_device->worker_core_from_logical_core(sender_worker_cores[link * 2]);

        // Writer
        std::vector<uint32_t> writer_forward_rt_args = {
            input_tensor_Wt,               // width in tiles of the input shard
            input_tensor_Ht,               // height in tiles of the input shard
            output_tensor_Wt,              // width in tiles of entire output
            output_tensor_Ht,              // height in tiles of entire output
            dim,                           // dim to gather on
            batch_head_size,               // product of the first two dims
            input_tile_id_start,           //
            input_tile_id_end,             //
            sender_forward_worker_core.x,  // out_ready_sem_noc0_x
            sender_forward_worker_core.y,  // out_ready_sem_noc0_y
            ring_size,                     // ring_size
            semaphore.at(1).address()      // out_ready_semaphore_backward
        };
        writer_sender_rt_offset = writer_forward_rt_args.size();
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            writer_forward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        writer_forward_rt_args.push_back(false);
        writer_forward_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            const auto target_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto backward_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id,
                backward_fabric_node_id,
                link,
                program,
                sender_worker_cores[(link * 2) + 1],
                writer_forward_rt_args);
        }
        if (fuse_op) {
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                writer_forward_rt_args, num_links, link, 1);
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_writer_forward_kernel_id,
            sender_worker_cores[(link * 2) + 1],
            writer_forward_rt_args);

        std::vector<uint32_t> writer_backward_rt_args = {
            input_tensor_Wt,                // width in tiles of the input shard
            input_tensor_Ht,                // height in tiles of the input shard
            output_tensor_Wt,               // width in tiles of entire output
            output_tensor_Ht,               // height in tiles of entire output
            dim,                            // dim to gather on
            batch_head_size,                // product of the first two dims
            input_tile_id_start,            //
            input_tile_id_end,              //
            sender_backward_worker_core.x,  // out_ready_sem_noc0_x
            sender_backward_worker_core.y,  // out_ready_sem_noc0_y
            ring_size,                      // ring_size
            semaphore.at(0).address()       // out_ready_semaphore_backward
        };
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            writer_backward_rt_args.push_back(output_tensor[input_idx].buffer()->address());
        }
        writer_backward_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            const auto target_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto forward_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id,
                forward_fabric_node_id,
                link,
                program,
                sender_worker_cores[link * 2],
                writer_backward_rt_args);
        }
        writer_backward_rt_args.push_back(false);
        if (fuse_op) {
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_backward_rt_args, 1, 0, 0);
        }
        tt::tt_metal::SetRuntimeArgs(
            program, worker_sender_writer_backward_kernel_id, sender_worker_cores[link * 2], writer_backward_rt_args);
    }

    return {
        worker_sender_reader_forward_kernel_id,
        worker_sender_writer_forward_kernel_id,
        worker_sender_reader_backward_kernel_id,
        worker_sender_writer_backward_kernel_id,
        sender_worker_cores,
        num_inputs,
        reader_sender_rt_offset,
        writer_sender_rt_offset,
        num_links};
}

void ring_attention_all_gather_async_multicore_with_workers_override_runtime_arguments(
    const RingAttentionAllGatherAsyncMultiCoreWithWorkersSharedVariables& shared_variables,
    Program& program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<Tensor>& output_tensors,
    const std::vector<GlobalSemaphore>& semaphore) {
    // Extract shared variables
    const auto& worker_sender_reader_forward_kernel_id = shared_variables.worker_sender_reader_forward_kernel_id;
    const auto& worker_sender_writer_forward_kernel_id = shared_variables.worker_sender_writer_forward_kernel_id;
    const auto& worker_sender_reader_backward_kernel_id = shared_variables.worker_sender_reader_backward_kernel_id;
    const auto& worker_sender_writer_backward_kernel_id = shared_variables.worker_sender_writer_backward_kernel_id;
    const auto& sender_worker_cores = shared_variables.sender_worker_cores;
    const auto& num_inputs = shared_variables.num_inputs;
    const auto& reader_sender_rt_offset = shared_variables.reader_sender_rt_offset;
    const auto& writer_sender_rt_offset = shared_variables.writer_sender_rt_offset;
    const auto& num_links = shared_variables.num_links;

    // update senders
    auto& worker_reader_sender_forward_runtime_args_by_core =
        GetRuntimeArgs(program, worker_sender_reader_forward_kernel_id);
    auto& worker_writer_sender_forward_runtime_args_by_core =
        GetRuntimeArgs(program, worker_sender_writer_forward_kernel_id);
    auto& worker_reader_sender_backward_runtime_args_by_core =
        GetRuntimeArgs(program, worker_sender_reader_backward_kernel_id);
    auto& worker_writer_sender_backward_runtime_args_by_core =
        GetRuntimeArgs(program, worker_sender_writer_backward_kernel_id);

    for (int link = 0; link < num_links; link++) {
        auto& worker_reader_sender_forward_runtime_args =
            worker_reader_sender_forward_runtime_args_by_core[sender_worker_cores[1 + (link * 2)].x]
                                                             [sender_worker_cores[1 + (link * 2)].y];
        auto& worker_reader_sender_backward_runtime_args =
            worker_reader_sender_backward_runtime_args_by_core[sender_worker_cores[0 + (link * 2)].x]
                                                              [sender_worker_cores[0 + (link * 2)].y];
        auto& worker_writer_sender_forward_runtime_args =
            worker_writer_sender_forward_runtime_args_by_core[sender_worker_cores[1 + (link * 2)].x]
                                                             [sender_worker_cores[1 + (link * 2)].y];
        auto& worker_writer_sender_backward_runtime_args =
            worker_writer_sender_backward_runtime_args_by_core[sender_worker_cores[0 + (link * 2)].x]
                                                              [sender_worker_cores[0 + (link * 2)].y];

        worker_reader_sender_forward_runtime_args[9] = semaphore.at(1).address();
        worker_reader_sender_backward_runtime_args[9] = semaphore.at(0).address();
        worker_writer_sender_forward_runtime_args[11] = semaphore.at(1).address();
        worker_writer_sender_backward_runtime_args[11] = semaphore.at(0).address();
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            // sender reader
            worker_reader_sender_forward_runtime_args[reader_sender_rt_offset + input_idx] =
                input_tensors[input_idx].buffer()->address();
            worker_reader_sender_forward_runtime_args[reader_sender_rt_offset + num_inputs + input_idx] =
                output_tensors[input_idx].buffer()->address();
            worker_reader_sender_backward_runtime_args[reader_sender_rt_offset + input_idx] =
                input_tensors[input_idx].buffer()->address();
            worker_reader_sender_backward_runtime_args[reader_sender_rt_offset + num_inputs + input_idx] =
                output_tensors[input_idx].buffer()->address();
            // sender writer
            worker_writer_sender_forward_runtime_args[writer_sender_rt_offset + input_idx] =
                output_tensors[input_idx].buffer()->address();
            worker_writer_sender_backward_runtime_args[writer_sender_rt_offset + input_idx] =
                output_tensors[input_idx].buffer()->address();
        }
    }
}

}  // namespace ttnn

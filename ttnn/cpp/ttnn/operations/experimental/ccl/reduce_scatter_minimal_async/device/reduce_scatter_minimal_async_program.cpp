// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
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
    const std::optional<uint32_t>& cluster_axis) {
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
        empty_fused_op_signaler);
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
            fused_op_signaler);
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
            fused_op_signaler);
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

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores
    // 2 sender (reader + core + writer), 1 forward 1 backward
    uint32_t num_senders_per_link = 2;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_senders_per_link, mesh_device, sub_device_id, core_grid_offset);
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto& input_tensor_shape = input_tensor.get_padded_shape();
    const auto intermediate_tensor_buffer_type = intermediate_tensor.buffer()->buffer_type();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto num_batches = input_tensor_shape[0];
    const auto batch_slice_num_pages = input_tensor_num_pages / ring_size / num_batches;
    const auto batch_slice_num_pages_per_link = batch_slice_num_pages / num_links;

    for (uint32_t i = 0; i < sender_worker_cores.size(); i++) {
        const auto& core = sender_worker_cores[i];
        if (i % 2 == 1) {
            sender_forward_core_ranges.insert(CoreRange(core));
        } else {
            sender_backward_core_ranges.insert(CoreRange(core));
        }
    }
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = num_tiles_to_write_per_packet < 4 ? 4 * num_tiles_to_write_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_input_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_intermediate_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_reader_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_compute_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_compute_output_config);

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
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    TT_FATAL(
        !(input_tensor_shape[3] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[3],
        tt::constants::TILE_WIDTH);
    uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;

    bool input_is_sharded = input_tensor.is_sharded();
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (intermediate_is_sharded) {
        reader_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
        writer_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    // KERNEL CREATION
    // Reader
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    for (uint32_t core_idx = 0; core_idx < num_senders_per_link; core_idx++) {
        std::vector<uint32_t> sender_reader_compile_args = {
            ring_index,                                              // my_chip_id
            static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
            static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
            input_cb_index,                                          // cb_input_id
            intermediate_cb_index,                                   // cb_intermediate_id
            reader_output_cb_index,                                  // cb_reader_output_id
            tile_granularity,                                        // packet_size_in_pages
            page_size,                                               // tensor0_page_size
            input_tensor_Wt,                                         // input_tensor_Wt
            batch_slice_num_pages,                                   // batch_slice_num_pages
            ring_size,                                               // ring_size
            num_batches,                                             // num_batches
            fuse_op,                                                 // fused op
            core_idx % num_senders_per_link,                         // direction
        };
        if (input_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
        }
        if (intermediate_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
        }
        auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
            "ring_reduce_scatter_minimal_async_reader.cpp",
            core_idx % num_senders_per_link ? sender_forward_core_ranges : sender_backward_core_ranges,
            tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));
        reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

        // Writer
        std::vector<uint32_t> sender_writer_compile_args = {
            ring_index,                                              // my_chip_id
            reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
            num_packet_headers_storable,                             // num_packet_headers_storable
            static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
            static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
            compute_output_cb_index,                                 // cb_compute_output_id
            reader_output_cb_index,                                  // cb_reader_output_id
            tile_granularity,                                        // packet_size_in_pages
            page_size,                                               // tensor0_page_size
            input_tensor_Wt,                                         // input_tensor_Wt
            batch_slice_num_pages,                                   // batch_slice_num_pages
            ring_size,                                               // ring_size
            num_batches,                                             // num_batches
            num_tiles_to_write_per_packet,                           // num_tiles_to_write_per_packet
            core_idx % num_senders_per_link,                         // direction
        };
        if (intermediate_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
        }
        if (output_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
        }
        auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
            "ring_reduce_scatter_minimal_async_writer.cpp",
            core_idx % num_senders_per_link ? sender_forward_core_ranges : sender_backward_core_ranges,
            tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));
        writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

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
            num_links,
            core_idx % num_senders_per_link};

        auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
            "ring_reduction.cpp",
            core_idx % num_senders_per_link ? sender_forward_core_ranges : sender_backward_core_ranges,
            sender_reduce_kernel_config);
        reduce_kernel_ids.push_back(sender_reduce_kernel_id);
    }

    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range);
    }

    // Kernel Runtime Args
    CoreCoord drain_sync_core;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t core_idx = 0; core_idx < num_senders_per_link; core_idx++) {
            CoreCoord core = sender_worker_cores[link * 2 + core_idx];
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),         // input_tensor_address
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                semaphore.at(core_idx).address(),         // out_ready_semaphore
                semaphore.at(2).address(),                // batch_ready_semaphore
                link,
                num_links,
                input_tensor_Wt / ring_size,                                                 // slice_Wt
                (link * batch_slice_num_pages / num_links) % (input_tensor_Wt / ring_size),  // start_pages_read_in_row
                (link * batch_slice_num_pages / num_links) / (input_tensor_Wt / ring_size) *
                    input_tensor_Wt,                            // start_row_offset
                link * batch_slice_num_pages / num_links,       // start_tiles_read
                (link + 1) * batch_slice_num_pages / num_links  // start_tiles_to_read
            };
            if (input_is_sharded) {
                shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
            }
            if (intermediate_is_sharded) {
                shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
            }
            if (fuse_op) {
                fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_ids[core_idx], {core}, reader_rt_args);

            std::vector<uint32_t> writer_rt_args = {
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                output_tensor.buffer()->address(),        // output_tensor_address
                drain_sync_core.x,                        // out_ready_sem_noc0_x
                drain_sync_core.y,                        // out_ready_sem_noc0_y
                semaphore.at(core_idx).address(),         // out_ready_fwd_semaphore
                semaphore.at(2).address(),                // batch_ready_semaphore
                link,
                num_links,
                input_tensor_Wt / ring_size,                                                 // slice_Wt
                (link * batch_slice_num_pages / num_links) % (input_tensor_Wt / ring_size),  // pages_read_in_row
                (link * batch_slice_num_pages / num_links) / (input_tensor_Wt / ring_size) *
                    input_tensor_Wt,                            // row_offset
                (link * batch_slice_num_pages / num_links),     // tiles_read
                (link + 1) * batch_slice_num_pages / num_links  // tiles_to_read
            };
            if (intermediate_is_sharded) {
                shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
            }
            if (output_is_sharded) {
                shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
            }
            if (core_idx % num_senders_per_link) {  // forward
                writer_rt_args.push_back(forward_device.has_value());
                if (forward_device.has_value()) {
                    const auto sender_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                    const auto forward_device_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        sender_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, writer_rt_args);
                }
                writer_rt_args.push_back(false);
            } else {
                writer_rt_args.push_back(false);
                writer_rt_args.push_back(backward_device.has_value());
                if (backward_device.has_value()) {
                    const auto sender_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                    const auto backward_device_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        sender_fabric_node_id, backward_device_fabric_node_id, link, program, {core}, writer_rt_args);
                }
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_ids[core_idx], {core}, writer_rt_args);

            std::vector<uint32_t> reduce_rt_args = {link};
            tt::tt_metal::SetRuntimeArgs(program, reduce_kernel_ids[core_idx], {core}, reduce_rt_args);
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_ids, writer_kernel_ids, sender_worker_cores, num_senders_per_link](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];

            // update senders
            std::vector<std::vector<std::vector<RuntimeArgsData>>> reader_runtime_args_by_core;
            std::vector<std::vector<std::vector<RuntimeArgsData>>> writer_runtime_args_by_core;
            for (uint32_t core_idx = 0; core_idx < num_senders_per_link; core_idx++) {
                reader_runtime_args_by_core.push_back(GetRuntimeArgs(program, reader_kernel_ids[core_idx]));
                writer_runtime_args_by_core.push_back(GetRuntimeArgs(program, writer_kernel_ids[core_idx]));
            }
            for (uint32_t i = 0; i < sender_worker_cores.size(); i++) {
                CoreCoord core = sender_worker_cores[i];
                // sender reader
                auto& worker_reader_sender_runtime_args =
                    reader_runtime_args_by_core[i % num_senders_per_link][core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                // sender writer
                auto& worker_writer_sender_runtime_args =
                    writer_runtime_args_by_core[i % num_senders_per_link][core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
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

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    bool fuse_op = fused_op_signaler.has_value();

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores
    // 2 sender (reader + core + writer), 1 forward 1 backward
    uint32_t num_senders_per_routing_plane = 2;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_senders_per_routing_plane, mesh_device, sub_device_id, core_grid_offset);
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;

    for (uint32_t i = 0; i < sender_worker_cores.size(); i++) {
        const auto& core = sender_worker_cores[i];
        if (i % 2 == 1) {
            sender_forward_core_ranges.insert(CoreRange(core));
        } else {
            sender_backward_core_ranges.insert(CoreRange(core));
        }
    }
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    const uint32_t max_scatter_write_pages = 2;
    const uint32_t max_dst_size = 8;  // TODO: generalize based on arch and fp32 acc
    uint32_t tiles_to_write_per_packet = std::min(num_pages_per_packet, max_scatter_write_pages);
    uint32_t tile_granularity = std::min(4 * num_pages_per_packet, max_dst_size);
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_input_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_intermediate_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_reader_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_compute_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_compute_output_config);

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
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto& input_tensor_shape = input_tensor.get_padded_shape();
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

    bool input_is_sharded = input_tensor.is_sharded();
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (intermediate_is_sharded) {
        reader_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
        writer_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        reader_compute_defines["OUTPUT_IS_SHARDED"] = "1";
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    // KERNEL CREATION
    // Reader
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    for (uint32_t core_idx = 0; core_idx < num_senders_per_routing_plane; core_idx++) {
        const bool is_forward = core_idx % num_senders_per_routing_plane;

        /**
         * Every chip has a final reduction step. On the ends, there is only one input to reduce.
         * In the middle, you must reduce from both input directions.
         * FWD/BWD readers need to synchronize in order to avoid race conditions.
         *
         * We'll say that FWD always leads, then signals BWD to follow.
         */

        const bool is_first_device_in_direction = is_forward ? is_first_chip : is_last_chip;
        const int num_targets_in_direction = is_forward ? num_targets_forward : num_targets_backward;
        // The number of reduction steps is 0 for chips on the ends, or num_targets_in_direction for chips in the middle
        const int num_intermediate_reduction_steps = is_first_device_in_direction ? 0 : num_targets_in_direction;
        const bool do_final_reduction = !is_first_device_in_direction;
        const int num_total_reduction_steps = num_intermediate_reduction_steps + (do_final_reduction ? 1 : 0);
        const bool sync_with_other_direction = !(is_first_chip || is_last_chip);

        // Reader
        std::vector<uint32_t> sender_reader_compile_args = {
            ring_index,                                              // my_chip_id
            static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
            static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
            static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
            input_cb_index,                                          // cb_input_id
            intermediate_cb_index,                                   // cb_intermediate_id
            reader_output_cb_index,                                  // cb_reader_output_id
            tile_granularity,                                        // packet_size_in_pages
            page_size,                                               // tensor0_page_size
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
        };
        if (input_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
        }
        if (intermediate_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
        }
        if (output_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(output_tensor, sender_reader_compile_args);
        }
        auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
            "line_reduce_scatter_minimal_async_reader.cpp",
            is_forward ? sender_forward_core_ranges : sender_backward_core_ranges,
            tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));
        reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

        // Writer
        std::vector<uint32_t> sender_writer_compile_args = {
            ring_index,                                              // my_chip_id
            reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
            num_packet_headers_storable,                             // num_packet_headers_storable
            static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
            static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
            compute_output_cb_index,                                 // cb_compute_output_id
            reader_output_cb_index,                                  // cb_reader_output_id
            tile_granularity,                                        // packet_size_in_pages
            page_size,                                               // tensor0_page_size
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
        };
        if (intermediate_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
        }
        if (output_is_sharded) {
            shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
        }
        auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
            "line_reduce_scatter_minimal_async_writer.cpp",
            is_forward ? sender_forward_core_ranges : sender_backward_core_ranges,
            tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));
        writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

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
            num_links,
            num_total_reduction_steps};

        auto reduce_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
            "line_reduction.cpp",
            is_forward ? sender_forward_core_ranges : sender_backward_core_ranges,
            sender_reduce_kernel_config);
        reduce_kernel_ids.push_back(reduce_kernel_id);
    }

    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range);
    }

    // Kernel Runtime Args
    uint32_t fwd_bwd_semaphore_address = tt::tt_metal::CreateSemaphore(program, sender_worker_core_range, 0);
    CoreCoord drain_sync_core;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t core_idx = 0; core_idx < num_senders_per_routing_plane; core_idx++) {
            const bool is_forward = core_idx % num_senders_per_routing_plane;

            CoreCoord core = sender_worker_cores[link * 2 + core_idx];
            // FWD core needs BWD core coordinate for fwd/bwd final reduction sync.
            // For final synchronization, each core needs to know the coordinate of the opposite direction's core.
            uint32_t opposite_core_idx = link * 2 + (1 - core_idx);
            auto opposite_core = sender_worker_cores[opposite_core_idx];
            auto opposite_core_coord = mesh_device->worker_core_from_logical_core(opposite_core);
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),         // input_tensor_address
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                output_tensor.buffer()->address(),        // output_tensor_address
                semaphore.at(0 + link * 3).address(),     // remote transfer sync semaphore
                link,
                num_links,
                fwd_bwd_semaphore_address};
            if (input_is_sharded) {
                shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
            }
            if (intermediate_is_sharded) {
                shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
            }
            if (output_is_sharded) {
                shard_builder::extend_sharding_run_time_args(output_tensor, reader_rt_args);
            }
            if (fuse_op) {
                fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_ids[core_idx], {core}, reader_rt_args);

            std::vector<uint32_t> writer_rt_args = {
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                output_tensor.buffer()->address(),        // output_tensor_address
                drain_sync_core.x,                        // out_ready_sem_noc0_x
                drain_sync_core.y,                        // out_ready_sem_noc0_y
                semaphore.at(0 + link * 3).address(),     // remote transfer sync semaphore
                semaphore.at(1 + link * 3).address(),     // final reduction slot semaphore
                semaphore.at(2 + link * 3).address(),     // batch_ready_semaphore
                link,
                num_links,
                fwd_bwd_semaphore_address,
                opposite_core_coord.x,
                opposite_core_coord.y};
            if (intermediate_is_sharded) {
                shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
            }
            if (output_is_sharded) {
                shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
            }
            if (core_idx % num_senders_per_routing_plane) {  // forward
                writer_rt_args.push_back(forward_device.has_value());
                if (forward_device.has_value()) {
                    const auto sender_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                    const auto forward_device_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        sender_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, writer_rt_args);
                }
                writer_rt_args.push_back(false);
            } else {
                writer_rt_args.push_back(false);
                writer_rt_args.push_back(backward_device.has_value());
                if (backward_device.has_value()) {
                    const auto sender_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                    const auto backward_device_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        sender_fabric_node_id, backward_device_fabric_node_id, link, program, {core}, writer_rt_args);
                }
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_ids[core_idx], {core}, writer_rt_args);

            std::vector<uint32_t> reduce_rt_args = {link};
            tt::tt_metal::SetRuntimeArgs(program, reduce_kernel_ids[core_idx], {core}, reduce_rt_args);
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_ids, writer_kernel_ids, sender_worker_cores, num_senders_per_routing_plane](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];

            // update senders
            std::vector<std::vector<std::vector<RuntimeArgsData>>> reader_runtime_args_by_core;
            std::vector<std::vector<std::vector<RuntimeArgsData>>> writer_runtime_args_by_core;
            for (uint32_t core_idx = 0; core_idx < num_senders_per_routing_plane; core_idx++) {
                reader_runtime_args_by_core.push_back(GetRuntimeArgs(program, reader_kernel_ids[core_idx]));
                writer_runtime_args_by_core.push_back(GetRuntimeArgs(program, writer_kernel_ids[core_idx]));
            }
            for (uint32_t i = 0; i < sender_worker_cores.size(); i++) {
                CoreCoord core = sender_worker_cores[i];
                // sender reader
                auto& worker_reader_sender_runtime_args =
                    reader_runtime_args_by_core[i % num_senders_per_routing_plane][core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                worker_reader_sender_runtime_args[2] = output.buffer()->address();
                // sender writer
                auto& worker_writer_sender_runtime_args =
                    writer_runtime_args_by_core[i % num_senders_per_routing_plane][core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace ttnn

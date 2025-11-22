// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/ring_reduce_scatter_minimal_async_program.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program_common.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

namespace {

std::vector<uint32_t> get_ring_reader_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const bool fuse_op,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            ring_index,               // my_chip_id
            ring_size,                // ring_size
            input_cb_index,           // cb_input_id
            intermediate_cb_index,    // cb_intermediate_id
            reader_output_cb_index,   // cb_reader_output_id
            tile_granularity,         // tile_granularity
            page_size,                // page_size
            output_tensor_num_pages,  // output_num_pages
            input_batch_num_pages,    // batch_num_pages
            slice_B,                  // slice_B
        };
    } else {
        return {
            ring_index,               // my_chip_id
            ring_size,                // ring_size
            input_cb_index,           // cb_input_id
            intermediate_cb_index,    // cb_intermediate_id
            reader_output_cb_index,   // cb_reader_output_id
            tile_granularity,         // tile_granularity
            page_size,                // page_size
            input_batch_num_pages,    // input_batch_num_pages
            input_channel_num_pages,  // input_channel_num_pages
            input_tensor_B,           // input_tensor_B
            input_tensor_Wt,          // input_tensor_Wt
            slice_C,                  // slice_C
            slice_Ht,                 // slice_Ht
            slice_Wt,                 // slice_Wt
            fuse_op,                  // fused op
            normalized_dim,           // dim normalized to 4D
        };
    }
}

std::vector<uint32_t> get_ring_writer_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t compute_output_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t num_tiles_to_write_per_packet,
    const uint32_t output_tensor_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            ring_index,                     // my_chip_id
            ring_size,                      // ring_size
            compute_output_cb_index,        // cb_compute_output_id
            reader_output_cb_index,         // cb_reader_output_id
            tile_granularity,               // packet_size_in_pages
            page_size,                      // page_size
            num_tiles_to_write_per_packet,  // num_tiles_to_write_per_packet
            output_tensor_num_pages,        // output_num_pages
            input_batch_num_pages,          // batch_num_pages
            slice_B,                        // slice_B
        };
    } else {
        return {
            ring_index,                     // my_chip_id
            ring_size,                      // ring_size
            compute_output_cb_index,        // cb_compute_output_id
            reader_output_cb_index,         // cb_reader_output_id
            tile_granularity,               // packet_size_in_pages
            page_size,                      // page_size
            num_tiles_to_write_per_packet,  // num_tiles_to_write_per_packet
            output_batch_num_pages,         // output_batch_num_pages
            input_channel_num_pages,        // input_channel_num_pages
            output_channel_num_pages,       // output_channel_num_pages
            input_tensor_B,                 // input_tensor_B
            input_tensor_Wt,                // input_tensor_Wt
            slice_C,                        // slice_C
            slice_Ht,                       // slice_Ht
            slice_Wt,                       // slice_Wt
            normalized_dim                  // dim normalized to 4D
        };
    }
}

std::vector<uint32_t> get_ring_reduce_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t ring_size,
    const uint32_t input_tensor_B,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            input_cb_index,           // input_cb_id
            intermediate_cb_index,    // intermediate_cb
            compute_output_cb_index,  // output_cb
            tile_granularity,         // tile_granularity
            ring_size,                // ring_size
            slice_B,                  // slice_B
        };
    } else {
        return {
            input_cb_index,           // input_cb_id
            intermediate_cb_index,    // intermediate_cb
            compute_output_cb_index,  // output_cb
            tile_granularity,         // tile_granularity
            ring_size,                // ring_size
            input_tensor_B,           // input_tensor_B
            slice_C,                  // slice_C
        };
    }
}

}  // anonymous namespace

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring {

void override_runtime_args(
    tt::tt_metal::Program& program,
    const ReduceScatterProgramArtifacts& artifacts,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output) {
    // update senders
    for (uint32_t link = 0; link < artifacts.num_links; link++) {
        for (uint32_t dir = 0; dir < artifacts.num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < artifacts.num_workers_per_direction; worker++) {
                uint32_t mux_core_offset =
                    (link * artifacts.num_cores_per_link) +
                    (dir * (artifacts.num_mux_cores_per_direction_per_link + artifacts.num_workers_per_direction));
                CoreCoord core =
                    artifacts.all_cores[mux_core_offset + artifacts.num_mux_cores_per_direction_per_link + worker];
                std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                    GetRuntimeArgs(program, artifacts.reader_kernel_id);
                std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                    GetRuntimeArgs(program, artifacts.writer_kernel_id);

                // sender reader
                auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                worker_reader_sender_runtime_args[2] = semaphore.at(dir).address();
                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
                worker_writer_sender_runtime_args[4] = semaphore.at(dir).address();
                worker_writer_sender_runtime_args[5] = semaphore.at(artifacts.num_directions_per_link).address();

                if (barrier_semaphore.has_value()) {
                    worker_writer_sender_runtime_args[7] = barrier_semaphore.value().address();
                }
            }
        }
    }
}

tt::tt_metal::operation::ProgramWithCallbacks populate_program_legacy(
    tt::tt_metal::Program& program,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t&
        operation_attributes,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t& tensor_args,
    Tensor& intermediate_tensor,
    Tensor& output_tensor,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t&
        mesh_runtime_params) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    auto mesh_device = input_tensor.device();
    const uint32_t ring_size = operation_attributes.ring_size;
    [[maybe_unused]] bool is_first_chip = mesh_runtime_params.ring_index == 0;
    [[maybe_unused]] bool is_last_chip = mesh_runtime_params.ring_index == ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        mesh_runtime_params.sender_device_coord,
        is_first_chip,
        is_last_chip);

    bool fuse_op = mesh_runtime_params.fused_op_signaler.has_value();

    // op hyperparams
    // Get worker cores
    // 2 senders per direction (2: forward, backward) per link (num_links)
    // Each sender is reader + compute + writer
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint32_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction = mesh_runtime_params.num_workers_per_direction_opt.value_or(
        operations::experimental::ccl::detail::default_workers(
            *mesh_device,
            operation_attributes.sub_device_id,
            operation_attributes.topology,
            input_data_size_bytes,
            operation_attributes.num_links,
            operation_attributes.ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link));
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = operation_attributes.num_buffers_per_channel.value_or(1);

    uint32_t num_cores_per_link =
        operations::experimental::ccl::detail::reduce_scatter_minimal_async_core_count_per_link(
            num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        operation_attributes.topology,
        mesh_runtime_params.sender_device_coord,
        mesh_runtime_params.forward_coord,
        mesh_runtime_params.backward_coord,
        mesh_device);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        operation_attributes.topology,
        mesh_runtime_params.sender_device_coord,
        mesh_runtime_params.forward_coord,
        mesh_runtime_params.backward_coord,
        ring_size - 1,
        ring_size - 1,
        mesh_device);

    const auto [all_core_range, all_cores] = ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_cores_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        mesh_runtime_params.core_grid_offset);

    std::vector<CoreRange> sender_worker_core_ranges;
    std::vector<CoreRange> mux_core_ranges;
    std::vector<CoreRange> termination_master_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            mux_core_ranges.emplace_back(mux_core);

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                sender_worker_core_ranges.emplace_back(worker_core);

                if (worker == 0) {
                    termination_master_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const auto [normalized_dim, input_tensor_C, input_tensor_B] =
        (input_tensor_shape.rank() == 2)
            ? operations::experimental::ccl::detail::map_2d_to_4d(operation_attributes.dim)
            : operations::experimental::ccl::detail::map_nd_to_4d(input_tensor_shape, operation_attributes.dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t slice_B = input_tensor_B;
    uint32_t slice_C = input_tensor_C;
    uint32_t slice_Ht = input_tensor_Ht;
    uint32_t slice_Wt = input_tensor_Wt;
    if (normalized_dim == 0) {
        slice_B /= ring_size;
    } else if (normalized_dim == 1) {
        slice_C /= ring_size;
    } else if (normalized_dim == 2) {
        slice_Ht /= ring_size;
    } else if (normalized_dim == 3) {
        slice_Wt /= ring_size;
    } else {
        TT_FATAL(
            false, "reduce_scatter_minimal_async ring implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async ring implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = num_tiles_to_write_per_packet < 4 ? 4 * num_tiles_to_write_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

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
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        mesh_runtime_params.fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const auto num_full_size_channels = num_workers_per_direction;
    constexpr auto num_header_only_channels = 0;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // Fabric mux kernel
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    std::vector<uint32_t> sender_reader_compile_args = get_ring_reader_compile_args(
        mesh_runtime_params.ring_index,
        ring_size,
        input_cb_index,
        intermediate_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        output_tensor_num_pages,
        input_batch_num_pages,
        input_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        fuse_op,
        normalized_dim);

    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_reader.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduce_scatter_minimal_async_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args = get_ring_writer_compile_args(
        mesh_runtime_params.ring_index,
        ring_size,
        compute_output_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        num_tiles_to_write_per_packet,
        output_tensor_num_pages,
        output_batch_num_pages,
        input_batch_num_pages,
        input_channel_num_pages,
        output_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        normalized_dim);

    ccl::append_fabric_mux_connection_ct_args(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        num_workers_per_direction,
        sender_writer_compile_args);

    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_writer_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }

    std::string sender_writer_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduce_scatter_minimal_async_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = get_ring_reduce_compile_args(
        input_cb_index,
        intermediate_cb_index,
        compute_output_cb_index,
        tile_granularity,
        ring_size,
        input_tensor_B,
        slice_B,
        slice_C,
        normalized_dim);

    std::string sender_reduce_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduction.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduction.cpp";

    auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            auto mux_logical_core = *((mux_core_iter++)->begin());
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

            std::vector<uint32_t> mux_rt_args = {};
            const auto src_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.sender_device_coord);
            if (dir) {  // forward
                const auto dst_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.forward_coord.value());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            } else {
                const auto dst_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.backward_coord.value());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            }
            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                uint32_t worker_id = (link * num_workers_per_direction) + worker;
                uint32_t num_workers = operation_attributes.num_links * num_workers_per_direction;

                auto [start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset] =
                    operations::experimental::ccl::detail::get_tile_offsets(
                        worker_id,
                        num_workers,
                        output_batch_num_pages,
                        output_channel_num_pages,
                        slice_Wt,
                        input_tensor_Wt,
                        normalized_dim);

                // for dim 0 scatters we process each slice in batches
                // for all other dims we process each slice in channels
                uint32_t tiles_to_process_per_slice =
                    (start_tiles_to_read - start_tiles_read) * (normalized_dim == 0 ? slice_B : slice_C);
                uint32_t chunks_per_sync_val = operation_attributes.chunks_per_sync.value_or(
                    operations::experimental::ccl::detail::default_chunks_per_sync(
                        operation_attributes.topology, tiles_to_process_per_slice, tile_granularity));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),                  // input_tensor_address
                    intermediate_tensor.buffer()->address(),           // intermediate_tensor_address
                    operation_attributes.semaphore.at(dir).address(),  // out_ready_semaphore
                    dir,                                               // direction
                    chunks_per_sync_val,                               // chunks_per_sync
                    start_tiles_read,                                  // start_tiles_read
                    start_tiles_to_read,                               // start_tiles_to_read
                    start_pages_read_in_row,                           // start_pages_read_in_row
                    start_row_offset,                                  // start_row_offset (unused by dim0 kernel)
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    mesh_runtime_params.fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer RT args
                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),           // intermediate_tensor_address
                    output_tensor.buffer()->address(),                 // output_tensor_address
                    virtual_core.x,                                    // out_ready_sem_noc0_x
                    virtual_core.y,                                    // out_ready_sem_noc0_y
                    operation_attributes.semaphore.at(dir).address(),  // out_ready_fwd_semaphore
                    operation_attributes.semaphore.at(num_directions_per_link).address(),  // batch_ready_semaphore
                    operation_attributes.barrier_semaphore.has_value() &&
                        !operation_attributes.using_persistent_buffers,  // use_barrier_sem
                    operation_attributes.barrier_semaphore.has_value()   // barrier_sem
                        ? operation_attributes.barrier_semaphore.value().address()
                        : 0,
                    dir,                      // direction
                    chunks_per_sync_val,      // chunks_per_sync
                    start_pages_read_in_row,  // start_pages_read_in_row (unused by dim0 kernel)
                    start_row_offset,         // start_row_offset (unused by dim0 kernel)
                    start_tiles_read,         // start_tiles_read
                    start_tiles_to_read,      // tiles_to_read

                };
                ccl::append_fabric_mux_connection_rt_args(
                    true,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_kernel_config,
                    core,
                    worker,
                    worker == 0,
                    termination_master_virtual_core,
                    program,
                    writer_rt_args);
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);

                std::vector<uint32_t> reduce_rt_args = {
                    start_tiles_read,     // start_tiles_read
                    start_tiles_to_read,  // start_tiles_to_read
                    dir};                 // dir
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    ReduceScatterProgramArtifacts shared_vars = {
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link,
        operation_attributes.num_links};

    auto override_runtime_arguments_callback =
        [shared_vars](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];
            // TODO: Update matmul_reduce_scatter_async to use new TMP device operation pattern
            // The old ReduceScatterMinimalAsync struct no longer exists. This callback needs to be
            // updated when matmul_reduce_scatter_async is migrated.
            TT_THROW(
                "populate_program_legacy callback is deprecated. "
                "Please migrate matmul_reduce_scatter_async to use the new TMP device operation pattern.");
            std::optional<GlobalSemaphore> barrier_semaphore = std::nullopt;
            std::vector<GlobalSemaphore> semaphore;
            override_runtime_args(program, shared_vars, barrier_semaphore, semaphore, input, intermed, output);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring {

typename RingReduceScatterMinimalAsyncProgramFactory::cached_mesh_workload_t
RingReduceScatterMinimalAsyncProgramFactory::create_mesh_workload(
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t&
        operation_attributes,
    const ::ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t& tensor_args,
    ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<::ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Verify tensor_return_value has both tensors
    TT_FATAL(
        tensor_return_value.size() >= 2,
        "tensor_return_value must contain both intermediate and output tensors. "
        "Expected size >= 2, got {}",
        tensor_return_value.size());

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(::ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

::ttnn::device_operation::CachedProgram<RingReduceScatterMinimalAsyncProgramFactory::shared_variables_t>
RingReduceScatterMinimalAsyncProgramFactory::create_at(
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t&
        operation_attributes,
    const ::ttnn::MeshCoordinate& mesh_coordinate,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t& tensor_args,
    ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};

    const std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);

    const std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor,
        mesh_coordinate,
        -1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");

    uint32_t device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    // Get intermediate tensor from tensor_return_value (index 0 is intermediate, index 1 is output)
    TT_FATAL(
        tensor_return_value.size() >= 2,
        "tensor_return_value must contain both intermediate and output tensors. "
        "Expected size >= 2, got {}",
        tensor_return_value.size());
    Tensor& intermediate_tensor = tensor_return_value[0];  // Non-const reference needed for build function

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> empty_fused_op_signaler;

    operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t mesh_runtime_params{
        .sender_device_coord = mesh_coordinate,
        .forward_coord = forward_coord,
        .backward_coord = backward_coord,
        .ring_index = device_index,
        .fused_op_signaler = empty_fused_op_signaler,  // non-const reference
        .num_workers_per_direction_opt = operation_attributes.num_workers_per_link,
        .core_grid_offset = CoreCoord(0, 0)};

    const Tensor& input_tensor = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value[1];  // Use output tensor (index 1)
    auto mesh_device = input_tensor.device();
    const uint32_t ring_size = operation_attributes.ring_size;
    [[maybe_unused]] bool is_first_chip = mesh_runtime_params.ring_index == 0;
    [[maybe_unused]] bool is_last_chip = mesh_runtime_params.ring_index == ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        mesh_runtime_params.sender_device_coord,
        is_first_chip,
        is_last_chip);

    bool fuse_op = mesh_runtime_params.fused_op_signaler.has_value();

    // op hyperparams
    // Get worker cores
    // 2 senders per direction (2: forward, backward) per link (num_links)
    // Each sender is reader + compute + writer
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint32_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction = mesh_runtime_params.num_workers_per_direction_opt.value_or(
        operations::experimental::ccl::detail::default_workers(
            *mesh_device,
            operation_attributes.sub_device_id,
            operation_attributes.topology,
            input_data_size_bytes,
            operation_attributes.num_links,
            operation_attributes.ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link));
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = operation_attributes.num_buffers_per_channel.value_or(1);

    uint32_t num_cores_per_link =
        operations::experimental::ccl::detail::reduce_scatter_minimal_async_core_count_per_link(
            num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        operation_attributes.topology,
        mesh_runtime_params.sender_device_coord,
        mesh_runtime_params.forward_coord,
        mesh_runtime_params.backward_coord,
        mesh_device);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        operation_attributes.topology,
        mesh_runtime_params.sender_device_coord,
        mesh_runtime_params.forward_coord,
        mesh_runtime_params.backward_coord,
        ring_size - 1,
        ring_size - 1,
        mesh_device);

    const auto [all_core_range, all_cores] = ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_cores_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        mesh_runtime_params.core_grid_offset);

    std::vector<CoreRange> sender_worker_core_ranges;
    std::vector<CoreRange> mux_core_ranges;
    std::vector<CoreRange> termination_master_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            mux_core_ranges.emplace_back(mux_core);

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                sender_worker_core_ranges.emplace_back(worker_core);

                if (worker == 0) {
                    termination_master_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const auto [normalized_dim, input_tensor_C, input_tensor_B] =
        (input_tensor_shape.rank() == 2)
            ? operations::experimental::ccl::detail::map_2d_to_4d(operation_attributes.dim)
            : operations::experimental::ccl::detail::map_nd_to_4d(input_tensor_shape, operation_attributes.dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t slice_B = input_tensor_B;
    uint32_t slice_C = input_tensor_C;
    uint32_t slice_Ht = input_tensor_Ht;
    uint32_t slice_Wt = input_tensor_Wt;
    if (normalized_dim == 0) {
        slice_B /= ring_size;
    } else if (normalized_dim == 1) {
        slice_C /= ring_size;
    } else if (normalized_dim == 2) {
        slice_Ht /= ring_size;
    } else if (normalized_dim == 3) {
        slice_Wt /= ring_size;
    } else {
        TT_FATAL(
            false, "reduce_scatter_minimal_async ring implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async ring implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = num_tiles_to_write_per_packet < 4 ? 4 * num_tiles_to_write_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

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
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        mesh_runtime_params.fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const auto num_full_size_channels = num_workers_per_direction;
    constexpr auto num_header_only_channels = 0;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // Fabric mux kernel
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    std::vector<uint32_t> sender_reader_compile_args = get_ring_reader_compile_args(
        mesh_runtime_params.ring_index,
        ring_size,
        input_cb_index,
        intermediate_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        output_tensor_num_pages,
        input_batch_num_pages,
        input_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        fuse_op,
        normalized_dim);

    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_reader.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduce_scatter_minimal_async_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args = get_ring_writer_compile_args(
        mesh_runtime_params.ring_index,
        ring_size,
        compute_output_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        num_tiles_to_write_per_packet,
        output_tensor_num_pages,
        output_batch_num_pages,
        input_batch_num_pages,
        input_channel_num_pages,
        output_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        normalized_dim);

    ccl::append_fabric_mux_connection_ct_args(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        num_workers_per_direction,
        sender_writer_compile_args);

    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_writer_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }

    std::string sender_writer_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduce_scatter_minimal_async_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = get_ring_reduce_compile_args(
        input_cb_index,
        intermediate_cb_index,
        compute_output_cb_index,
        tile_granularity,
        ring_size,
        input_tensor_B,
        slice_B,
        slice_C,
        normalized_dim);

    std::string sender_reduce_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduction.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduction.cpp";

    auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            auto mux_logical_core = *((mux_core_iter++)->begin());
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

            std::vector<uint32_t> mux_rt_args = {};
            const auto src_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.sender_device_coord);
            if (dir) {  // forward
                const auto dst_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.forward_coord.value());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            } else {
                const auto dst_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.backward_coord.value());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            }
            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                uint32_t worker_id = (link * num_workers_per_direction) + worker;
                uint32_t num_workers = operation_attributes.num_links * num_workers_per_direction;

                auto [start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset] =
                    operations::experimental::ccl::detail::get_tile_offsets(
                        worker_id,
                        num_workers,
                        output_batch_num_pages,
                        output_channel_num_pages,
                        slice_Wt,
                        input_tensor_Wt,
                        normalized_dim);

                // for dim 0 scatters we process each slice in batches
                // for all other dims we process each slice in channels
                uint32_t tiles_to_process_per_slice =
                    (start_tiles_to_read - start_tiles_read) * (normalized_dim == 0 ? slice_B : slice_C);
                uint32_t chunks_per_sync_val = operation_attributes.chunks_per_sync.value_or(
                    operations::experimental::ccl::detail::default_chunks_per_sync(
                        operation_attributes.topology, tiles_to_process_per_slice, tile_granularity));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),                  // input_tensor_address
                    intermediate_tensor.buffer()->address(),           // intermediate_tensor_address
                    operation_attributes.semaphore.at(dir).address(),  // out_ready_semaphore
                    dir,                                               // direction
                    chunks_per_sync_val,                               // chunks_per_sync
                    start_tiles_read,                                  // start_tiles_read
                    start_tiles_to_read,                               // start_tiles_to_read
                    start_pages_read_in_row,                           // start_pages_read_in_row
                    start_row_offset,                                  // start_row_offset (unused by dim0 kernel)
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    mesh_runtime_params.fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer RT args
                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),           // intermediate_tensor_address
                    output_tensor.buffer()->address(),                 // output_tensor_address
                    virtual_core.x,                                    // out_ready_sem_noc0_x
                    virtual_core.y,                                    // out_ready_sem_noc0_y
                    operation_attributes.semaphore.at(dir).address(),  // out_ready_fwd_semaphore
                    operation_attributes.semaphore.at(num_directions_per_link).address(),  // batch_ready_semaphore
                    operation_attributes.barrier_semaphore.has_value() &&
                        !operation_attributes.using_persistent_buffers,  // use_barrier_sem
                    operation_attributes.barrier_semaphore.has_value()   // barrier_sem
                        ? operation_attributes.barrier_semaphore.value().address()
                        : 0,
                    dir,                      // direction
                    chunks_per_sync_val,      // chunks_per_sync
                    start_pages_read_in_row,  // start_pages_read_in_row (unused by dim0 kernel)
                    start_row_offset,         // start_row_offset (unused by dim0 kernel)
                    start_tiles_read,         // start_tiles_read
                    start_tiles_to_read,      // tiles_to_read

                };
                ccl::append_fabric_mux_connection_rt_args(
                    true,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_kernel_config,
                    core,
                    worker,
                    worker == 0,
                    termination_master_virtual_core,
                    program,
                    writer_rt_args);
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);

                std::vector<uint32_t> reduce_rt_args = {
                    start_tiles_read,     // start_tiles_read
                    start_tiles_to_read,  // start_tiles_to_read
                    dir};                 // dir
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    shared_variables_t shared_vars = {
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link,
        operation_attributes.num_links};

    return {std::move(program), std::move(shared_vars)};
}

void RingReduceScatterMinimalAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Get intermediate tensor from tensor_return_value (index 0 is intermediate, index 1 is output)
    TT_FATAL(
        tensor_return_value.size() >= 2,
        "tensor_return_value must contain both intermediate and output tensors. "
        "Expected size >= 2, got {}",
        tensor_return_value.size());
    const Tensor& intermediate_tensor = tensor_return_value[0];

    auto& programs = cached_workload.workload.get_programs();
    for (auto& [range, shared_vars] : cached_workload.shared_variables) {
        auto& program = programs.at(range);
        override_runtime_args(
            program,
            shared_vars,
            operation_attributes.barrier_semaphore,
            operation_attributes.semaphore,
            tensor_args.input_tensor,
            intermediate_tensor,
            tensor_return_value[1]);  // Use output tensor (index 1)
    }
}

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring

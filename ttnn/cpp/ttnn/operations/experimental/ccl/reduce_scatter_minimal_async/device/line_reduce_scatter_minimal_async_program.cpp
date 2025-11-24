// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/line_reduce_scatter_minimal_async_program.hpp"

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

struct LineReduceScatterConfig {
    bool is_first_chip;
    bool is_last_chip;
    bool fuse_op;
    uint32_t num_directions_per_link;
    uint32_t num_mux_cores_per_direction_per_link;
    uint32_t num_workers_per_direction;
    uint32_t num_buffers_full_size_channels;
    uint32_t num_cores_per_link;
    uint32_t page_size;
    std::tuple<std::array<uint32_t, 2>, std::array<uint32_t, 2>> unicast_args;
    std::tuple<uint32_t, uint32_t> num_targets;
    std::tuple<std::array<uint32_t, 6>, std::array<uint32_t, 6>> mcast_args;
};

struct LineReduceScatterCoreAllocation {
    std::vector<CoreCoord> all_cores;
    CoreRangeSet sender_worker_core_range_set;
    CoreRangeSet mux_core_range_set;
    std::vector<CoreRange> termination_master_core_ranges;
    std::function<bool(uint32_t)> mux_connection_valid;
};

struct LineReduceScatterTensorInfo {
    uint32_t normalized_dim;
    uint32_t input_tensor_C;
    uint32_t input_tensor_B;
    uint32_t input_tensor_Ht;
    uint32_t input_tensor_Wt;
    uint32_t slice_B;
    uint32_t slice_C;
    uint32_t slice_Ht;
    uint32_t slice_Wt;
    uint32_t input_tensor_num_pages;
    uint32_t output_tensor_num_pages;
    uint32_t input_batch_num_pages;
    uint32_t output_batch_num_pages;
    uint32_t input_channel_num_pages;
    uint32_t output_channel_num_pages;
    bool input_is_sharded;
    bool intermediate_is_sharded;
    bool output_is_sharded;
    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;
};

struct LineReduceScatterCircularBuffers {
    uint32_t input_cb_index;
    uint32_t intermediate_cb_index;
    uint32_t reader_output_cb_index;
    uint32_t compute_output_cb_index;
    uint32_t tile_granularity{};
    uint32_t tiles_to_write_per_packet{};
    uint32_t cb_num_pages{};
    tt::DataFormat df{};
};

struct LineReduceScatterKernelArgs {
    tt::tt_metal::Program& program;
    const Tensor& input_tensor;
    const Tensor& intermediate_tensor;
    const Tensor& output_tensor;
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes;
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params;
    const LineReduceScatterConfig& config;
    const LineReduceScatterCoreAllocation& core_allocation;
    const LineReduceScatterCircularBuffers& cbs;
    const LineReduceScatterTensorInfo& tensor_info;
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config;
    const bool sync_with_other_direction;
};

struct MuxKernelInfo {
    tt::tt_metal::KernelHandle mux_kernel_id{};
    tt::tt_fabric::FabricMuxConfig mux_kernel_config;
    uint32_t fwd_bwd_semaphore_address{};
};

struct KernelInfo {
    MuxKernelInfo mux;
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle reduce_kernel_id{};
};

LineReduceScatterConfig setup_line_reduce_scatter_configuration(
    const Tensor& input_tensor,
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params) {
    const uint32_t ring_size = operation_attributes.ring_size;
    auto mesh_device = input_tensor.device();
    bool is_first_chip = mesh_runtime_params.ring_index == 0;
    bool is_last_chip = mesh_runtime_params.ring_index == ring_size - 1;
    bool fuse_op = mesh_runtime_params.fused_op_signaler.has_value();

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
            ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link));
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = operation_attributes.num_buffers_per_channel.value_or(1);

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        mesh_runtime_params.sender_device_coord,
        is_first_chip,
        is_last_chip);

    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        operation_attributes.topology,
        mesh_runtime_params.sender_device_coord,
        mesh_runtime_params.forward_coord,
        mesh_runtime_params.backward_coord,
        mesh_device);
    auto [num_targets_forward, num_targets_backward] = ccl::get_forward_backward_line_mcast_distance(
        ring_size, mesh_runtime_params.ring_index, operation_attributes.topology, true);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        operation_attributes.topology,
        mesh_runtime_params.sender_device_coord,
        mesh_runtime_params.forward_coord,
        mesh_runtime_params.backward_coord,
        num_targets_forward,
        num_targets_backward,
        mesh_device);

    uint32_t num_cores_per_link =
        operations::experimental::ccl::detail::reduce_scatter_minimal_async_core_count_per_link(
            num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    return {
        is_first_chip,
        is_last_chip,
        fuse_op,
        num_directions_per_link,
        num_mux_cores_per_direction_per_link,
        num_workers_per_direction,
        num_buffers_full_size_channels,
        num_cores_per_link,
        page_size,
        std::make_tuple(unicast_forward_args, unicast_backward_args),
        std::make_tuple(num_targets_forward, num_targets_backward),
        std::make_tuple(mcast_forward_args, mcast_backward_args)};
}

LineReduceScatterCoreAllocation allocate_line_reduce_scatter_cores(
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params,
    const LineReduceScatterConfig& config,
    const Tensor& input_tensor) {
    auto mesh_device = input_tensor.device();
    const auto [all_core_range, all_cores] = ccl::choose_worker_cores(
        operation_attributes.num_links,
        config.num_cores_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        mesh_runtime_params.core_grid_offset);

    const auto mux_connection_valid = [&mesh_runtime_params](const uint32_t dir) {
        return (!dir && mesh_runtime_params.backward_coord.has_value()) ||
               (dir && mesh_runtime_params.forward_coord.has_value());
    };

    std::vector<CoreRange> sender_worker_core_ranges;
    std::vector<CoreRange> mux_core_ranges;
    std::vector<CoreRange> termination_master_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        for (uint32_t dir = 0; dir < config.num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (mux_connection_valid(dir)) {
                mux_core_ranges.emplace_back(mux_core);
            }
            for (uint32_t worker = 0; worker < config.num_workers_per_direction; worker++) {
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

    return {
        all_cores,
        sender_worker_core_range_set,
        mux_core_range_set,
        termination_master_core_ranges,
        mux_connection_valid};
}

LineReduceScatterCircularBuffers create_line_reduce_scatter_circular_buffers(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const LineReduceScatterConfig& config,
    const LineReduceScatterCoreAllocation& core_allocation) {
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = config.page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t max_scatter_write_pages = 2;
    const uint32_t max_dst_size = 8;  // TODO: generalize based on arch and fp32 acc
    uint32_t tiles_to_write_per_packet = std::min(num_pages_per_packet, max_scatter_write_pages);
    uint32_t tile_granularity = std::min(4 * num_pages_per_packet, max_dst_size);
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    auto create_circular_buffer = [&](uint32_t cb_index) -> uint32_t {
        tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{cb_index, df}})
                .set_page_size(cb_index, l1_scratch_cb_page_size_bytes);
        CreateCircularBuffer(program, core_allocation.sender_worker_core_range_set, cb_config);
        return cb_index;
    };

    uint32_t input_cb_index = create_circular_buffer(tt::CB::c_in0);
    uint32_t intermediate_cb_index = create_circular_buffer(tt::CB::c_in1);
    uint32_t reader_output_cb_index = create_circular_buffer(tt::CB::c_in2);
    uint32_t compute_output_cb_index = create_circular_buffer(tt::CB::c_in3);

    return {
        input_cb_index,
        intermediate_cb_index,
        reader_output_cb_index,
        compute_output_cb_index,
        tile_granularity,
        tiles_to_write_per_packet,
        cb_num_pages,
        df};
}

LineReduceScatterTensorInfo calculate_line_reduce_scatter_tensor_info(
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const Tensor& output_tensor,
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const LineReduceScatterConfig& config) {
    const uint32_t ring_size = operation_attributes.ring_size;
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
            false, "reduce_scatter_minimal_async line implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(config.fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async line implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

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

    return {
        normalized_dim,
        input_tensor_C,
        input_tensor_B,
        input_tensor_Ht,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        input_tensor_num_pages,
        output_tensor_num_pages,
        input_batch_num_pages,
        output_batch_num_pages,
        input_channel_num_pages,
        output_channel_num_pages,
        input_is_sharded,
        intermediate_is_sharded,
        output_is_sharded,
        reader_compute_defines,
        writer_compute_defines};
}

MuxKernelInfo create_mux_kernel(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const LineReduceScatterConfig& config,
    const LineReduceScatterCoreAllocation& core_allocation) {
    uint32_t fwd_bwd_semaphore_address =
        tt::tt_metal::CreateSemaphore(program, core_allocation.sender_worker_core_range_set, 0);
    auto mesh_device = input_tensor.device();
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;

    const auto num_full_size_channels = config.num_workers_per_direction;
    const auto num_header_only_channels = 0;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    tt::tt_fabric::FabricMuxConfig mux_kernel_config(
        num_full_size_channels,
        num_header_only_channels,
        config.num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        core_allocation.mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    return {mux_kernel_id, mux_kernel_config, fwd_bwd_semaphore_address};
}

tt::tt_metal::KernelHandle create_reader_kernel(const LineReduceScatterKernelArgs& args) {
    auto get_line_reader_compile_args = [&]() -> std::vector<uint32_t> {
        if (args.tensor_info.normalized_dim == 0) {
            return {
                args.mesh_runtime_params.ring_index,       // my_chip_id
                args.operation_attributes.ring_size,       // ring_size
                args.cbs.input_cb_index,                   // cb_input_id
                args.cbs.intermediate_cb_index,            // cb_intermediate_id
                args.cbs.reader_output_cb_index,           // cb_reader_output_id
                args.cbs.tile_granularity,                 // tile_granularity
                args.config.page_size,                     // page_size
                args.tensor_info.input_tensor_num_pages,   // input_num_pages
                args.tensor_info.output_tensor_num_pages,  // output_num_pages
                args.tensor_info.input_batch_num_pages,    // batch_num_pages
                args.tensor_info.slice_B,                  // slice_B
                args.sync_with_other_direction             // sync_with_other_direction
            };
        } else {
            return {
                args.mesh_runtime_params.ring_index,        // my_chip_id
                args.operation_attributes.ring_size,        // ring_size
                args.cbs.input_cb_index,                    // cb_input_id
                args.cbs.intermediate_cb_index,             // cb_intermediate_id
                args.cbs.reader_output_cb_index,            // cb_reader_output_id
                args.cbs.tile_granularity,                  // tile_granularity
                args.config.page_size,                      // page_size
                args.tensor_info.input_tensor_num_pages,    // input_num_pages
                args.tensor_info.input_batch_num_pages,     // input_batch_num_pages
                args.tensor_info.input_channel_num_pages,   // input_channel_num_pages
                args.tensor_info.output_batch_num_pages,    // output_batch_num_pages
                args.tensor_info.output_channel_num_pages,  // output_channel_num_pages
                args.tensor_info.input_tensor_B,            // input_tensor_B
                args.tensor_info.input_tensor_Wt,           // input_tensor_Wt
                args.tensor_info.slice_C,                   // slice_C
                args.tensor_info.slice_Ht,                  // slice_Ht
                args.tensor_info.slice_Wt,                  // slice_Wt
                args.config.fuse_op,                        // fuse_op
                args.sync_with_other_direction,             // sync_with_other_direction
                args.tensor_info.normalized_dim,            // dim
            };
        }
    };

    std::vector<uint32_t> sender_reader_compile_args = get_line_reader_compile_args();

    if (args.tensor_info.input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(args.input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(args.input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (args.tensor_info.intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(args.intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(args.intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (args.tensor_info.output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(args.output_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(args.output_tensor.buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        args.tensor_info.normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                               "device/kernels/dim_zero_line_reduce_scatter_minimal_async_reader.cpp"
                                             : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                               "device/kernels/line_reduce_scatter_minimal_async_reader.cpp";

    return tt::tt_metal::CreateKernel(
        args.program,
        sender_reader_kernel_path,
        args.core_allocation.sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, args.tensor_info.reader_compute_defines));
}

tt::tt_metal::KernelHandle create_writer_kernel(const LineReduceScatterKernelArgs& args) {
    auto [unicast_forward_args, unicast_backward_args] = args.config.unicast_args;
    auto [mcast_forward_args, mcast_backward_args] = args.config.mcast_args;

    auto get_line_writer_compile_args = [&]() -> std::vector<uint32_t> {
        if (args.tensor_info.normalized_dim == 0) {
            return {
                args.operation_attributes.ring_size,       // ring_size
                args.cbs.compute_output_cb_index,          // cb_compute_output_id
                args.cbs.reader_output_cb_index,           // cb_reader_output_id
                args.cbs.tile_granularity,                 // tile_granularity
                args.config.page_size,                     // page_size
                args.cbs.tiles_to_write_per_packet,        // contig_pages_advanced
                args.tensor_info.input_tensor_num_pages,   // input_num_pages
                args.tensor_info.output_tensor_num_pages,  // output_num_pages
                args.tensor_info.input_batch_num_pages,    // batch_num_pages
                args.tensor_info.slice_B,                  // slice_B
                args.sync_with_other_direction,            // sync_with_other_direction
            };
        } else {
            return {
                args.operation_attributes.ring_size,        // ring_size
                args.cbs.compute_output_cb_index,           // cb_compute_output_id
                args.cbs.reader_output_cb_index,            // cb_reader_output_id
                args.cbs.tile_granularity,                  // tile_granularity
                args.config.page_size,                      // page_size
                args.cbs.tiles_to_write_per_packet,         // contig_pages_advanced
                args.tensor_info.input_tensor_num_pages,    // input_num_pages
                args.tensor_info.input_batch_num_pages,     // input_batch_num_pages
                args.tensor_info.input_channel_num_pages,   // input_channel_num_pages
                args.tensor_info.output_batch_num_pages,    // output_batch_num_pages
                args.tensor_info.output_channel_num_pages,  // output_channel_num_pages
                args.tensor_info.input_tensor_B,            // input_tensor_b
                args.tensor_info.input_tensor_Wt,           // input_tensor_Wt
                args.tensor_info.slice_C,                   // slice_C
                args.tensor_info.slice_Ht,                  // slice_Ht
                args.tensor_info.slice_Wt,                  // slice_Wt
                args.tensor_info.normalized_dim,            // dim
                args.sync_with_other_direction              // sync_with_other_direction
            };
        }
    };

    std::vector<uint32_t> compile_args = get_line_writer_compile_args();

    ccl::append_fabric_mux_connection_ct_args(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        args.mux_kernel_config,
        args.config.num_workers_per_direction,
        compile_args);

    compile_args.insert(compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    compile_args.insert(compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    compile_args.insert(compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    compile_args.insert(compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    auto append_tensor_args = [&](const Tensor& tensor, bool is_sharded) {
        if (is_sharded) {
            shard_builder::extend_sharding_compile_time_args(tensor, compile_args);
        } else {
            tt::tt_metal::TensorAccessorArgs(tensor.buffer()).append_to(compile_args);
        }
    };

    append_tensor_args(args.intermediate_tensor, args.tensor_info.intermediate_is_sharded);
    append_tensor_args(args.output_tensor, args.tensor_info.output_is_sharded);

    std::string sender_writer_kernel_path =
        args.tensor_info.normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                               "device/kernels/dim_zero_line_reduce_scatter_minimal_async_writer.cpp"
                                             : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                               "device/kernels/line_reduce_scatter_minimal_async_writer.cpp";

    return tt::tt_metal::CreateKernel(
        args.program,
        sender_writer_kernel_path,
        args.core_allocation.sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(compile_args, args.tensor_info.writer_compute_defines));
}

tt::tt_metal::KernelHandle create_compute_kernel(const LineReduceScatterKernelArgs& args) {
    auto get_line_reduce_compile_args = [&]() -> std::vector<uint32_t> {
        if (args.tensor_info.normalized_dim == 0) {
            return {
                args.cbs.input_cb_index,
                args.cbs.intermediate_cb_index,
                args.cbs.compute_output_cb_index,
                args.cbs.tile_granularity,
                args.tensor_info.slice_B};
        } else {
            return {
                args.cbs.input_cb_index,
                args.cbs.intermediate_cb_index,
                args.cbs.compute_output_cb_index,
                args.cbs.tile_granularity,
                args.tensor_info.input_tensor_B,
                args.tensor_info.slice_C};
        }
    };

    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = get_line_reduce_compile_args();

    std::string sender_reduce_kernel_path =
        args.tensor_info.normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                               "device/kernels/dim_zero_line_reduction.cpp"
                                             : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                               "device/kernels/line_reduction.cpp";

    return tt::tt_metal::CreateKernel(
        args.program,
        sender_reduce_kernel_path,
        args.core_allocation.sender_worker_core_range_set,
        sender_reduce_kernel_config);
}

KernelInfo create_kernels(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const Tensor& output_tensor,
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params,
    const LineReduceScatterConfig& config,
    const LineReduceScatterCoreAllocation& core_allocation,
    const LineReduceScatterCircularBuffers& cbs,
    const LineReduceScatterTensorInfo& tensor_info) {
    auto mesh_device = input_tensor.device();

    if (config.fuse_op) {
        mesh_runtime_params.fused_op_signaler->init_reduce_scatter(
            program, mesh_device, core_allocation.sender_worker_core_range_set);
    }

    auto mux_config = create_mux_kernel(program, input_tensor, config, core_allocation);

    const bool sync_with_other_direction = !(config.is_first_chip || config.is_last_chip);

    LineReduceScatterKernelArgs kernel_args{
        .program = program,
        .input_tensor = input_tensor,
        .intermediate_tensor = intermediate_tensor,
        .output_tensor = output_tensor,
        .operation_attributes = operation_attributes,
        .mesh_runtime_params = mesh_runtime_params,
        .config = config,
        .core_allocation = core_allocation,
        .cbs = cbs,
        .tensor_info = tensor_info,
        .mux_kernel_config = mux_config.mux_kernel_config,
        .sync_with_other_direction = sync_with_other_direction};

    auto reader_kernel_id = create_reader_kernel(kernel_args);
    auto writer_kernel_id = create_writer_kernel(kernel_args);
    auto reduce_kernel_id = create_compute_kernel(kernel_args);

    return KernelInfo{
        .mux = mux_config,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .reduce_kernel_id = reduce_kernel_id};
}

CoreCoord setup_mux_runtime_args_for_direction(
    tt::tt_metal::Program& program,
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params,
    const KernelInfo& kernels,
    uint32_t dir,
    uint32_t link,
    CoreCoord mux_logical_core,
    decltype(std::declval<Tensor>().device()) mesh_device) {
    CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
    std::vector<uint32_t> mux_rt_args = {};
    const auto src_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.sender_device_coord);
    if (dir) {  // forward
        const auto dst_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.forward_coord.value());
        mux_rt_args = kernels.mux.mux_kernel_config.get_fabric_mux_run_time_args(
            src_node_id, dst_node_id, link, program, {mux_logical_core});
    } else {
        const auto dst_node_id = mesh_device->get_fabric_node_id(mesh_runtime_params.backward_coord.value());
        mux_rt_args = kernels.mux.mux_kernel_config.get_fabric_mux_run_time_args(
            src_node_id, dst_node_id, link, program, {mux_logical_core});
    }
    tt::tt_metal::SetRuntimeArgs(program, kernels.mux.mux_kernel_id, {mux_logical_core}, mux_rt_args);
    return mux_virtual_core;
}

void setup_worker_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const Tensor& output_tensor,
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params,
    const LineReduceScatterConfig& config,
    const LineReduceScatterCoreAllocation& core_allocation,
    const LineReduceScatterTensorInfo& tensor_info,
    const LineReduceScatterCircularBuffers& cbs,
    const KernelInfo& kernels,
    uint32_t link,
    uint32_t dir,
    uint32_t worker,
    bool is_forward,
    int num_targets_forward,
    int num_targets_backward,
    CoreCoord mux_virtual_core,
    CoreCoord termination_master_virtual_core,
    CoreCoord core,
    decltype(std::declval<Tensor>().device()) mesh_device) {
    CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

    uint32_t opposite_mux_core_offset =
        (link * config.num_cores_per_link) +
        ((1 - dir) * (config.num_mux_cores_per_direction_per_link + config.num_workers_per_direction));
    uint32_t opposite_core_idx = opposite_mux_core_offset + config.num_mux_cores_per_direction_per_link + worker;
    auto opposite_core = core_allocation.all_cores[opposite_core_idx];
    auto opposite_core_coord = mesh_device->worker_core_from_logical_core(opposite_core);

    const bool is_first_device_in_direction = is_forward ? config.is_first_chip : config.is_last_chip;
    const int num_targets_in_direction = is_forward ? num_targets_forward : num_targets_backward;
    const int num_intermediate_reduction_steps = is_first_device_in_direction ? 0 : num_targets_in_direction;
    const bool do_final_reduction = !is_first_device_in_direction;
    const int num_total_reduction_steps = num_intermediate_reduction_steps + (do_final_reduction ? 1 : 0);

    const uint32_t worker_id = (link * config.num_workers_per_direction) + worker;
    const uint32_t num_workers = operation_attributes.num_links * config.num_workers_per_direction;
    const auto [start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset] =
        operations::experimental::ccl::detail::get_tile_offsets(
            worker_id,
            num_workers,
            tensor_info.output_batch_num_pages,
            tensor_info.output_channel_num_pages,
            tensor_info.slice_Wt,
            tensor_info.input_tensor_Wt,
            tensor_info.normalized_dim);

    uint32_t tiles_to_process_per_slice = (start_tiles_to_read - start_tiles_read) *
                                          (tensor_info.normalized_dim == 0 ? tensor_info.slice_B : tensor_info.slice_C);
    uint32_t chunks_per_sync_val =
        operation_attributes.chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
            operation_attributes.topology, tiles_to_process_per_slice, cbs.tile_granularity));
    log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

    std::vector<uint32_t> reader_rt_args = {
        input_tensor.buffer()->address(),
        intermediate_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        operation_attributes.semaphore.at(0).address(),
        kernels.mux.fwd_bwd_semaphore_address,
        is_forward,
        is_first_device_in_direction,
        num_targets_in_direction,
        do_final_reduction,
        chunks_per_sync_val,
        start_tiles_read,
        start_tiles_to_read,
        start_pages_read_in_row,
        start_row_offset,
    };

    if (tensor_info.input_is_sharded) {
        shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
    }
    if (tensor_info.intermediate_is_sharded) {
        shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
    }
    if (tensor_info.output_is_sharded) {
        shard_builder::extend_sharding_run_time_args(output_tensor, reader_rt_args);
    }
    if (config.fuse_op) {
        mesh_runtime_params.fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
    }
    tt::tt_metal::SetRuntimeArgs(program, kernels.reader_kernel_id, {core}, reader_rt_args);

    std::vector<uint32_t> writer_rt_args = {
        intermediate_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        virtual_core.x,
        virtual_core.y,
        operation_attributes.semaphore.at(0).address(),
        kernels.mux.fwd_bwd_semaphore_address,
        opposite_core_coord.x,
        opposite_core_coord.y,
        operation_attributes.barrier_semaphore.has_value() && !operation_attributes.using_persistent_buffers,
        operation_attributes.barrier_semaphore.has_value() ? operation_attributes.barrier_semaphore.value().address()
                                                           : 0,
        is_forward,
        is_first_device_in_direction,
        num_targets_in_direction,
        do_final_reduction,
        chunks_per_sync_val,
        start_pages_read_in_row,
        start_row_offset,
        start_tiles_read,
        start_tiles_to_read,
    };
    ccl::append_fabric_mux_connection_rt_args(
        core_allocation.mux_connection_valid(dir),
        mux_virtual_core,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        kernels.mux.mux_kernel_config,
        core,
        worker,
        worker == 0,
        termination_master_virtual_core,
        program,
        writer_rt_args);
    if (tensor_info.intermediate_is_sharded) {
        shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
    }
    if (tensor_info.output_is_sharded) {
        shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
    }

    tt::tt_metal::SetRuntimeArgs(program, kernels.writer_kernel_id, {core}, writer_rt_args);

    std::vector<uint32_t> reduce_rt_args = {
        num_total_reduction_steps,
        start_tiles_read,
        start_tiles_to_read,
    };
    tt::tt_metal::SetRuntimeArgs(program, kernels.reduce_kernel_id, {core}, reduce_rt_args);
}

void setup_line_reduce_scatter_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const Tensor& output_tensor,
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t& mesh_runtime_params,
    const LineReduceScatterConfig& config,
    const LineReduceScatterCoreAllocation& core_allocation,
    const LineReduceScatterTensorInfo& tensor_info,
    const LineReduceScatterCircularBuffers& cbs,
    const KernelInfo& kernels) {
    auto mesh_device = input_tensor.device();
    auto [num_targets_forward, num_targets_backward] = config.num_targets;

    auto worker_core_iter = core_allocation.sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = core_allocation.mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = core_allocation.termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        for (uint32_t dir = 0; dir < config.num_directions_per_link; dir++) {
            const bool is_forward = dir;
            CoreCoord mux_virtual_core = {0, 0};
            if (core_allocation.mux_connection_valid(dir)) {
                auto mux_logical_core = *((mux_core_iter++)->begin());
                mux_virtual_core = setup_mux_runtime_args_for_direction(
                    program, mesh_runtime_params, kernels, dir, link, mux_logical_core, mesh_device);
            }

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            CoreCoord termination_master_virtual_core =
                mesh_device->worker_core_from_logical_core(termination_master_logical_core);

            for (uint32_t worker = 0; worker < config.num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                setup_worker_runtime_args(
                    program,
                    input_tensor,
                    intermediate_tensor,
                    output_tensor,
                    operation_attributes,
                    mesh_runtime_params,
                    config,
                    core_allocation,
                    tensor_info,
                    cbs,
                    kernels,
                    link,
                    dir,
                    worker,
                    is_forward,
                    num_targets_forward,
                    num_targets_backward,
                    mux_virtual_core,
                    termination_master_virtual_core,
                    core,
                    mesh_device);
            }
        }
    }
}

struct MeshRuntimeParamsAndIntermediate {
    operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t mesh_runtime_params;
    Tensor& intermediate_tensor;
};

MeshRuntimeParamsAndIntermediate prepare_mesh_runtime_params(
    const operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t& tensor_args,
    operations::experimental::ccl::reduce_scatter_minimal_async::tensor_return_value_t& tensor_return_value) {
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
        .fused_op_signaler = empty_fused_op_signaler,
        .num_workers_per_direction_opt = operation_attributes.num_workers_per_link,
        .core_grid_offset = CoreCoord(0, 0)};

    return {mesh_runtime_params, intermediate_tensor};
}

}  // anonymous namespace

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::line {

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
                worker_reader_sender_runtime_args[2] = output.buffer()->address();
                worker_reader_sender_runtime_args[3] = semaphore.at(0).address();
                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
                worker_writer_sender_runtime_args[4] = semaphore.at(0).address();

                if (barrier_semaphore.has_value()) {
                    worker_writer_sender_runtime_args[9] = barrier_semaphore.value().address();
                }
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::line

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::line {

typename LineReduceScatterMinimalAsyncProgramFactory::cached_mesh_workload_t
LineReduceScatterMinimalAsyncProgramFactory::create_mesh_workload(
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

::ttnn::device_operation::CachedProgram<LineReduceScatterMinimalAsyncProgramFactory::shared_variables_t>
LineReduceScatterMinimalAsyncProgramFactory::create_at(
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t&
        operation_attributes,
    const ::ttnn::MeshCoordinate& mesh_coordinate,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t& tensor_args,
    ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};

    auto [mesh_runtime_params, intermediate_tensor] =
        prepare_mesh_runtime_params(operation_attributes, mesh_coordinate, tensor_args, tensor_return_value);

    const Tensor& input_tensor = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value[1];  // Use output tensor (index 1)
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
    const auto config =
        setup_line_reduce_scatter_configuration(input_tensor, operation_attributes, mesh_runtime_params);
    const auto core_allocation =
        allocate_line_reduce_scatter_cores(operation_attributes, mesh_runtime_params, config, input_tensor);
    const auto cbs = create_line_reduce_scatter_circular_buffers(program, input_tensor, config, core_allocation);
    const auto tensor_info = calculate_line_reduce_scatter_tensor_info(
        input_tensor, intermediate_tensor, output_tensor, operation_attributes, config);
    const auto kernels = create_kernels(
        program,
        input_tensor,
        intermediate_tensor,
        output_tensor,
        operation_attributes,
        mesh_runtime_params,
        config,
        core_allocation,
        cbs,
        tensor_info);
    setup_line_reduce_scatter_runtime_args(
        program,
        input_tensor,
        intermediate_tensor,
        output_tensor,
        operation_attributes,
        mesh_runtime_params,
        config,
        core_allocation,
        tensor_info,
        cbs,
        kernels);

    shared_variables_t shared_vars = {
        kernels.reader_kernel_id,
        kernels.writer_kernel_id,
        core_allocation.all_cores,
        config.num_directions_per_link,
        config.num_workers_per_direction,
        config.num_mux_cores_per_direction_per_link,
        config.num_cores_per_link,
        operation_attributes.num_links};

    return {std::move(program), std::move(shared_vars)};
}

void LineReduceScatterMinimalAsyncProgramFactory::override_runtime_arguments(
    typename LineReduceScatterMinimalAsyncProgramFactory::cached_mesh_workload_t& cached_workload,
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

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::line

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_multi_core_nd_shard_input_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeMultiCoreNDShardInputProgramFactory::cached_program_t UntilizeMultiCoreNDShardInputProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& output) {
    tt::tt_metal::Program program{};

    const auto& a = tensor_args.input;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    TT_FATAL(output.is_allocated(), "Output buffer should be allocated on device!");
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t output_tensor_width = output.padded_shape()[-1];
    uint32_t output_tensor_height = output.physical_volume() / output_tensor_width;
    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_input_row = tensor_width / tile_width;

    const auto& nd_shard_spec = a.nd_shard_spec().value();
    uint32_t input_shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t input_shard_width = nd_shard_spec.shard_shape[-1];

    const auto distribution_spec = a.buffer()->buffer_distribution_spec().value();

    uint32_t num_shards = distribution_spec.num_shards();
    const auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& groups = distribution_spec.core_groups();
    const auto& ordered_cores_with_data = distribution_spec.cores_with_data();
    uint32_t num_compute_cores = ordered_cores_with_data.size();
    const auto& compute_core_range = CoreRangeSet(tt::stl::Span<const CoreCoord>(ordered_cores_with_data));

    uint32_t num_tiles_per_input_block = input_shard_width / tile_width;
    uint32_t num_blocks_per_shard_plane =
        input_shard_height /
        tile_height;  // Note: a "shard plane" here refers to a 2D plane the size of the last 2 dimensions of the shard.
                      // For example, a shard of shape [b, c, h, w] has b * c planes each of shape [h, w].
    const auto& shard_shape = nd_shard_spec.shard_shape;
    size_t num_planes_per_shard = 1;
    if (shard_shape.rank() > 2) {
        for (int i = 0; i < static_cast<int>(shard_shape.rank()) - 2; ++i) {
            num_planes_per_shard *= shard_shape[i];
        }
    }
    uint32_t num_blocks_per_shard = num_planes_per_shard * num_blocks_per_shard_plane;
    uint32_t num_input_blocks_per_full_core = groups.num_shards_per_core_in_group_1 * num_blocks_per_shard;

    // Input CB
    uint32_t input_cb_num_tiles;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        input_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        input_cb_num_tiles = num_tiles_per_input_block * 2;
    }
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        compute_core_range,
        input_single_tile_size,
        input_cb_num_tiles,
        input_cb_data_format);

    // Output CB
    uint32_t output_cb_num_tiles;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        output_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        output_cb_num_tiles = num_tiles_per_input_block * 2;
    }
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        compute_core_range,
        output_single_tile_size,
        output_cb_num_tiles,
        output_cb_data_format);

    // Reader compile-time args and kernel
    KernelHandle unary_reader_kernel_id;
    // Sharded input
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src0_cb_index,
        (uint32_t)num_tiles_per_input_block,
        (uint32_t)num_shards,
        (uint32_t)num_compute_cores};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_nd_sharded.cpp",
        compute_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile-time args
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width =
        output_tensor_width;  // In height-sharded and interleaved cases, the output page is the entire tensor row
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        if (output.shard_spec().has_value()) {
            output_page_width = output.shard_spec().value().shape[1];
        } else {
            output_page_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        output_num_blocks_across_width = tt::div_up(output_tensor_width, output_page_width);
    }

    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = output_page_width;
    uint32_t output_stick_size = num_cols_per_output_block * output_element_size;
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_cb_index,
        (uint32_t)output_stick_size,
        (uint32_t)tile_height,
        (uint32_t)num_tiles_per_input_block,
        (uint32_t)output_num_blocks_across_width,
        (uint32_t)output_element_size,
        (uint32_t)num_cols_per_input_block,
        (uint32_t)num_cols_per_output_block,
        (uint32_t)input_single_tile_size,
        (uint32_t)num_shards,
        (uint32_t)num_compute_cores,
        (uint32_t)num_tiles_per_input_row,
        (uint32_t)tile_width,
        (uint32_t)output_tensor_width,
        (uint32_t)output_tensor_height,
    };

    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    TensorAccessorArgs(*src0_buffer)
        .append_to(writer_compile_time_args);  // For ND sharded input, we need info on the input buffer distribution

    // Writer kernel
    std::string writer_kernel_file =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp";

    KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        writer_kernel_file,
        compute_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute kernel file
    std::string compute_kernel;
    if (!use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp");
        unpack_to_dest_mode[src0_cb_index] =
            UnpackToDestMode::Default;  // TODO: We need SFPU untilize for FP32 (#30400, #33795)
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "pack_untilize_variable_num_blocks.cpp");
    }

    // Compute compile-time args and kernel
    // Note: This condition is always true for sharded input
    KernelHandle untilize_kernel_id = 0;
    std::map<std::string, std::string> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }
    if (!compute_core_range.ranges().empty()) {
        std::vector<uint32_t> compute_compile_time_args = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        untilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            compute_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_compile_time_args,
                .defines = compute_kernel_defines});
    }

    // Run-time args
    // Logic for ND sharding makes as few assumptions about page locations as possible. Padded pages will be handled
    // in the writer kernel.
    const auto& mapped_cores = page_mapping.all_cores;

    // Use page_mapping to count non-padding blocks per core
    // page_mapping.core_host_page_indices[core_id] contains host page indices for all device pages on that core,
    // with UncompressedBufferPageMapping::PADDING indicating padding pages
    uint32_t start_shard_id = 0;
    for (auto core : ordered_cores_with_data) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_input_blocks_to_process = 0;

        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const auto& host_page_indices = page_mapping.core_host_page_indices[core_idx];

            // Iterate through device pages in blocks of num_tiles_per_input_block.
            uint32_t page_offset = 0;
            const uint32_t total_pages = host_page_indices.size();

            while (page_offset < total_pages) {
                if (host_page_indices[page_offset] != UncompressedBufferPageMapping::PADDING) {
                    num_input_blocks_to_process++;
                } else if (page_offset == 0) {  // First page is PADDING means this core has no shards, no need to
                                                // iterate further. This should never happen, as we are iterating over
                                                // only cores with data.
                    break;
                }
                // Advance by num_tiles_per_input_block
                page_offset += num_tiles_per_input_block;
            }
        }
        // Reader run-time args
        std::vector<uint32_t> reader_run_time_args = {src0_buffer->address(), start_shard_id};

        // Writer run-time args
        std::vector<uint32_t> writer_run_time_args = {dst_buffer->address(), src0_buffer->address(), start_shard_id};
        start_shard_id++;

        // Compute run-time args
        std::vector<uint32_t> compute_run_time_args = {num_input_blocks_to_process};
        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, untilize_kernel_id, core, compute_run_time_args);
    }

    std::vector<CoreCoord> cores_with_run_time_args;
    cores_with_run_time_args.reserve(ordered_cores_with_data.size());
    cores_with_run_time_args.insert(
        cores_with_run_time_args.end(), ordered_cores_with_data.begin(), ordered_cores_with_data.end());

    return cached_program_t{
        std::move(program),
        {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output, cores_with_run_time_args}};
}

void UntilizeMultiCoreNDShardInputProgramFactory::override_runtime_arguments(
    UntilizeMultiCoreNDShardInputProgramFactory::cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores_with_runtime_args = cached_program.shared_variables.cores_with_runtime_args;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    // Reader and Writer update buffer addresses
    auto& runtime_args_by_core_reader = GetRuntimeArgs(program, reader_kernel_id);
    auto& runtime_args_by_core_writer = GetRuntimeArgs(program, writer_kernel_id);
    for (const CoreCoord& core : cores_with_runtime_args) {
        auto& runtime_args_reader = runtime_args_by_core_reader[core.x][core.y];
        runtime_args_reader[0] = src_buffer->address();
        auto& runtime_args_writer = runtime_args_by_core_writer[core.x][core.y];
        runtime_args_writer[0] = dst_buffer->address();
        runtime_args_writer[1] = src_buffer->address();
    }
}
}  // namespace ttnn::prim

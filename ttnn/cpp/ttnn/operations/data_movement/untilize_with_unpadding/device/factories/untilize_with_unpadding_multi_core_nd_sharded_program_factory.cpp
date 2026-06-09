// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_with_unpadding_multi_core_nd_sharded_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    ProgramDescriptor desc;

    // const auto& a = input;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t tensor_width = input.padded_shape()[-1];
    uint32_t output_tensor_width = output.padded_shape()[-1];
    uint32_t output_tensor_height = output.padded_shape()[-2];

    const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_input_row = tensor_width / tile_width;
    uint32_t num_tiles_per_output_row = tt::div_up(output_tensor_width, tile_width);

    const auto& nd_shard_spec = input.nd_shard_spec().value();
    uint32_t input_shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t input_shard_width = nd_shard_spec.shard_shape[-1];

    const auto distribution_spec = input.buffer()->buffer_distribution_spec().value();

    uint32_t num_shards = distribution_spec.num_shards();
    const auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& groups = distribution_spec.core_groups();
    const auto& ordered_cores_with_data = distribution_spec.cores_with_data();
    uint32_t num_compute_cores = ordered_cores_with_data.size();
    const auto& compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

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
    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_num_tiles * input_single_tile_size,
        .core_ranges = compute_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    // Output CB
    uint32_t output_cb_num_tiles;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        output_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        output_cb_num_tiles = num_tiles_per_input_block * 2;
    }
    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_num_tiles * output_single_tile_size,
        .core_ranges = compute_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // Reader compile-time args and kernel (sharded input)
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src0_cb_index,
        (uint32_t)num_tiles_per_input_block,
        (uint32_t)num_shards,
        (uint32_t)num_compute_cores};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = compute_core_range;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer compile-time args
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width =
        output_tensor_width;  // In height-sharded and interleaved cases, the output page is the entire tensor row
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
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
        (uint32_t)num_tiles_per_output_row,
        (uint32_t)tile_width,
        (uint32_t)output_tensor_width,
        (uint32_t)output_tensor_height,
        (uint32_t)input.padded_shape().rank()

    };
    std::vector<uint32_t>
        writer_common_runtime_args;  // Due to tensor squeezing from ND to 4D when the input tensor has rank > 4,
                                     // writer_common_runtime_args will have at most 8 entries.
    for (const auto dim : output.padded_shape()) {
        writer_common_runtime_args.push_back(dim);
    }
    for (const auto dim : input.padded_shape()) {
        writer_common_runtime_args.push_back(dim);
    }

    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    TensorAccessorArgs(*src0_buffer)
        .append_to(writer_compile_time_args);  // For ND sharded input, we need info on the input buffer distribution

    // Writer kernel
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = compute_core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = std::move(writer_common_runtime_args);

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute kernel file
    const std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp");

    // Compute compile-time args and kernel
    // Note: This condition is always true for sharded input
    KernelDescriptor::Defines compute_kernel_defines;
    if (input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 || input.dtype() == DataType::FLOAT32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    KernelDescriptor compute_desc;
    bool has_compute = !compute_core_range.ranges().empty();
    if (has_compute) {
        std::vector<uint32_t> compute_compile_time_args = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        compute_desc.kernel_source = compute_kernel;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = compute_core_range;
        compute_desc.compile_time_args = std::move(compute_compile_time_args);
        compute_desc.defines = std::move(compute_kernel_defines);
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
    }

    // Run-time args
    // Logic for ND sharding makes as few assumptions about page locations as possible. Padded pages will be handled
    // in the writer kernel.
    const auto& mapped_cores = page_mapping.all_cores;

    // Use page_mapping to count non-padding blocks per core
    // page_mapping.core_host_page_indices[core_id] contains host page indices for all device pages on that core,
    // with UncompressedBufferPageMapping::PADDING indicating padding pages
    uint32_t start_shard_id = 0;
    reader_desc.runtime_args.reserve(ordered_cores_with_data.size());
    writer_desc.runtime_args.reserve(ordered_cores_with_data.size());
    if (has_compute) {
        compute_desc.runtime_args.reserve(ordered_cores_with_data.size());
    }
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
        reader_desc.emplace_runtime_args(core, {src0_buffer, start_shard_id});

        // Writer run-time args
        writer_desc.emplace_runtime_args(core, {dst_buffer, src0_buffer, start_shard_id});
        start_shard_id++;

        // Compute run-time args
        if (has_compute) {
            compute_desc.emplace_runtime_args(core, {num_input_blocks_to_process});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (has_compute) {
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}
}  // namespace ttnn::prim

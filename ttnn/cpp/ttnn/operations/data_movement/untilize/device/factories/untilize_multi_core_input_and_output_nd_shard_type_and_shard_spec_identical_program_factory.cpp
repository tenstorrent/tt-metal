// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.hpp"

#include "ttnn/common/constants.hpp"

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory::create_descriptor(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    const auto& nd_shard_spec = a.nd_shard_spec().value();
    uint32_t shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t shard_width = nd_shard_spec.shard_shape[-1];
    CoreRangeSet grid = nd_shard_spec.grid;
    ShardOrientation orientation = nd_shard_spec.orientation;

    uint32_t shard_vol = nd_shard_spec.shard_shape.volume();

    uint32_t num_tiles_per_block = shard_width / tile_width;
    uint32_t num_blocks_per_shard = (shard_height / tile_height) * (shard_vol / (shard_height * shard_width));
    uint32_t num_tiles_per_shard = num_tiles_per_block * num_blocks_per_shard;

    const auto& distribution_spec = a.buffer()->buffer_distribution_spec().value();

    uint32_t total_shards = distribution_spec.num_shards();
    uint32_t num_cores = grid.num_cores();
    const auto& groups = distribution_spec.core_groups();
    uint32_t num_shards_per_core = groups.num_shards_per_core_in_group_1;
    log_debug(
        tt::LogOp,
        "ND sharding: total_shards={}, cores={}, base_shards_per_core={}",
        total_shards,
        num_cores,
        num_shards_per_core);

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;

    ProgramDescriptor desc;

    // Sharded input CB — globally allocated to the input buffer; framework patches
    // the CB address on cache hits via cb.buffer.
    {
        CBDescriptor cb_src0;
        cb_src0.total_size = num_tiles_per_shard * num_shards_per_core * input_single_tile_size;
        cb_src0.core_ranges = grid;
        cb_src0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        });
        cb_src0.buffer = src0_buffer;
        desc.cbs.push_back(std::move(cb_src0));
    }

    // Sharded output CB — globally allocated to the output buffer.
    {
        CBDescriptor cb_output;
        cb_output.total_size = num_tiles_per_shard * num_shards_per_core * output_single_tile_size;
        cb_output.core_ranges = grid;
        cb_output.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        });
        cb_output.buffer = dst_buffer;
        desc.cbs.push_back(std::move(cb_output));
    }

    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index};
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = grid;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = grid;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_compile_time_args = {num_tiles_per_block, src0_cb_index, output_cb_index};

    std::vector<std::pair<std::string, std::string>> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = grid;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.defines = std::move(compute_kernel_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // Run-time args
    auto cores = corerange_to_cores(grid, std::nullopt, orientation == ShardOrientation::ROW_MAJOR);
    auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& mapped_cores = page_mapping.all_cores;
    for (const auto& core : cores) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_blocks_to_process = 0;
        uint32_t num_tiles_to_process = 0;
        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const size_t num_shards_on_core = distribution_spec.num_shards_per_core(core_idx);
            num_blocks_to_process = num_blocks_per_shard * num_shards_on_core;
            num_tiles_to_process = num_tiles_per_block * num_blocks_to_process;
        }

        // Sharded readers/writers consume only the (per-launch) tile count; no Buffer* slot is
        // needed because the CB itself carries the buffer binding.
        reader_desc.emplace_runtime_args(core, {num_tiles_to_process});
        compute_desc.emplace_runtime_args(core, {num_blocks_to_process});
        writer_desc.emplace_runtime_args(core, {num_tiles_to_process});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim

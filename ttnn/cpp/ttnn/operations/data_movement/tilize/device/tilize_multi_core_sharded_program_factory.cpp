// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TilizeMultiCoreShardedProgramFactory::create_descriptor(
    const TilizeParams& /*operation_attributes*/, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    const CoreRangeSet& all_cores = shard_spec.grid;

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    // Sharded input CB — globally allocated to the input buffer; framework patches
    // the CB address on cache hits via cb.buffer.
    {
        CBDescriptor cb_src0;
        cb_src0.total_size = num_tiles_per_shard * input_single_tile_size;
        cb_src0.core_ranges = all_cores;
        cb_src0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        });
        cb_src0.buffer = src_buffer;
        desc.cbs.push_back(std::move(cb_src0));
    }

    // Sharded output CB — globally allocated to the output buffer.
    {
        CBDescriptor cb_output;
        cb_output.total_size = num_tiles_per_shard * output_single_tile_size;
        cb_output.core_ranges = all_cores;
        cb_output.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        });
        cb_output.buffer = dst_buffer;
        desc.cbs.push_back(std::move(cb_output));
    }

    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Sharded readers/writers consume only the (constant per-launch) num_tiles_per_shard
    // count; no Buffer* slot is needed because the CB itself carries the buffer binding.
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_desc.emplace_runtime_args(core, {num_tiles_per_shard});
        writer_desc.emplace_runtime_args(core, {num_tiles_per_shard});
    }

    std::vector<uint32_t> compute_args = {
        uint32_t(num_tiles_per_shard / num_tiles_per_row), uint32_t(num_tiles_per_row)};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/kernel/compute/tilize.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_llk_acc,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim

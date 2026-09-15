// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_multi_program_factory.hpp"

#include <algorithm>
#include <numeric>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

namespace {

uint32_t find_greatest_common_page_size(std::vector<uint32_t>& stick_sizes, uint32_t alignment) {
    TT_FATAL(!stick_sizes.empty(), "Need at least one stick size to find page size");
    uint32_t page_size = tt::align(stick_sizes[0], alignment);
    for (size_t idx = 1; idx < stick_sizes.size(); idx++) {
        const uint32_t padded_stick_size = tt::align(stick_sizes[idx], alignment);
        page_size = std::gcd(page_size, padded_stick_size);
    }
    return page_size;
}

}  // namespace

tt::tt_metal::ProgramDescriptor ConcatS2SMultiProgramFactory::create_descriptor(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const uint32_t rank = input_tensors[0].logical_shape().rank();
    const bool height_concat = is_height_concat(rank, operation_attributes.dim);
    // Height concat has to interleave per leading index (see num_leading_blocks). Width concat
    // does not: it appends along the stick, so one block is both correct and what the kernel did
    // before -- and for a height-sharded width concat a per-core block count is not even
    // well-defined, since a core's rows can straddle leading indices. Pinning it to 1 there keeps
    // width concat untouched by construction.
    const uint32_t num_blocks = height_concat ? num_leading_blocks(input_tensors[0]) : 1u;
    ProgramDescriptor desc;

    const uint32_t num_input_tensors = input_tensors.size();
    const uint32_t cb_dst_id = 16;
    TT_FATAL(num_input_tensors <= cb_dst_id, "Not enough circular buffer for {} inputs.", num_input_tensors);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    // Assume inputs and output have the same element size and alignment.
    const uint32_t element_size = input_tensors[0].element_size();
    const uint32_t alignment = input_tensors[0].buffer()->alignment();

    uint32_t page_size;
    uint32_t elements_per_page_width;
    uint32_t elements_per_page_height;
    if (rm_layout) {
        std::vector<uint32_t> all_stick_sizes;
        all_stick_sizes.reserve(input_tensors.size() + 1);
        all_stick_sizes.push_back(output.shard_spec().value().shape[1]);
        std::transform(
            input_tensors.begin(), input_tensors.end(), std::back_inserter(all_stick_sizes), [](const Tensor& tensor) {
                return tensor.element_size() * tensor.shard_spec().value().shape[1];
            });
        page_size = find_greatest_common_page_size(all_stick_sizes, alignment);
        elements_per_page_width = page_size / element_size;
        elements_per_page_height = 1;
    } else {
        page_size = tt::tile_size(cb_data_format);
        elements_per_page_width = TILE_WIDTH;
        elements_per_page_height = TILE_HEIGHT;
    }

    std::vector<uint32_t> input_num_pages_per_stick;
    std::vector<uint32_t> input_num_sticks;
    std::vector<uint32_t> input_write_offsets;
    std::vector<uint32_t> input_num_sticks_per_block;
    std::vector<uint32_t> input_block_strides;
    input_num_pages_per_stick.reserve(num_input_tensors);
    input_num_sticks.reserve(num_input_tensors);
    input_write_offsets.reserve(num_input_tensors);
    input_num_sticks_per_block.reserve(num_input_tensors);
    input_block_strides.reserve(num_input_tensors);

    // Assume inputs and output have the same sharding grid.
    const auto all_cores = input_tensors[0].shard_spec().value().grid;

    // Input CBs
    //
    // For height concat the write offset carries only this input's *prefix within one leading
    // block* -- block b of input i starts b * output_block_stride further on, and the kernel adds
    // that term. Advancing by the whole shard here, as this did before #55342, is what laid the
    // result out as [all of input 0; all of input 1; ...] instead of interleaving per leading
    // index. At one block the two are the same expression, so width concat and the rank-4
    // (1, 1, H, W) case are unchanged.
    uint32_t curr_input_write_offset = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_pages_per_stick.push_back(tt::div_up(shard_spec.shape[1], elements_per_page_width));
        input_num_sticks.push_back(tt::div_up(shard_spec.shape[0], elements_per_page_height));
        input_write_offsets.push_back(curr_input_write_offset);

        // A width-sharded shard spans the whole flattened height, so every leading index
        // contributes the same number of rows. Anything else is a shard spec this factory cannot
        // describe as a strided copy. Vacuous at num_blocks == 1.
        TT_FATAL(
            input_num_sticks[input_id] % num_blocks == 0,
            "Height concat: input {} has {} shard rows, which does not divide into the {} leading "
            "indices of shape {}.",
            input_id,
            input_num_sticks[input_id],
            num_blocks,
            input_tensors[input_id].padded_shape());
        input_num_sticks_per_block.push_back(input_num_sticks[input_id] / num_blocks);
        input_block_strides.push_back(
            page_size * input_num_pages_per_stick[input_id] * input_num_sticks_per_block[input_id]);

        const uint32_t input_num_pages = input_num_pages_per_stick[input_id] * input_num_sticks[input_id];
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size * input_num_pages,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_id),
                .data_format = cb_data_format,
                .page_size = page_size,
            }}},
            .buffer = input_tensors[input_id].buffer(),
        });

        // Height concat: this input's rows within one block. Width concat: one stick.
        // input_block_strides is page_size * input_num_pages when num_blocks == 1.
        curr_input_write_offset +=
            height_concat ? input_block_strides[input_id] : page_size * input_num_pages_per_stick[input_id];
    }

    // Output CB
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_num_pages_per_stick = tt::div_up(output_shard_spec.shape[1], elements_per_page_width);
    const uint32_t output_num_sticks = tt::div_up(output_shard_spec.shape[0], elements_per_page_height);
    desc.cbs.push_back(CBDescriptor{
        .total_size = page_size * output_num_sticks * output_num_pages_per_stick,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_dst_id),
            .data_format = cb_data_format,
            .page_size = page_size,
        }}},
        .buffer = output.buffer(),
    });

    const uint32_t output_stride = page_size * output_num_pages_per_stick;

    TT_FATAL(
        output_num_sticks % num_blocks == 0,
        "Height concat: output has {} shard rows, which does not divide into the {} leading "
        "indices of shape {}.",
        output_num_sticks,
        num_blocks,
        output.padded_shape());
    // The write offsets above were accumulated in units of each input's own pages-per-stick, but
    // they index the output shard, so for height concat the two have to agree. They do -- the
    // inputs differ only in the concat dim, which is not the width -- and the old code relied on
    // the same identity. Checked rather than assumed: a mismatch would skew every row silently.
    if (height_concat) {
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            TT_FATAL(
                input_num_pages_per_stick[input_id] == output_num_pages_per_stick,
                "Height concat: input {} is {} pages wide but the output is {}; height concat "
                "requires equal widths.",
                input_id,
                input_num_pages_per_stick[input_id],
                output_num_pages_per_stick);
        }
    }
    // Rows one leading index contributes to the output shard, in bytes.
    const uint32_t output_block_stride = output_stride * (output_num_sticks / num_blocks);

    const KernelDescriptor::CompileTimeArgs compile_time_args = {
        cb_dst_id, page_size, output_stride, num_input_tensors, num_blocks, output_block_stride};

    std::vector<uint32_t> runtime_args_0;
    std::vector<uint32_t> runtime_args_1;
    runtime_args_0.reserve(num_input_tensors * 5);
    runtime_args_1.reserve(num_input_tensors * 5);
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        // Split this input's rows *within one block* across the two RISCs, and let each RISC walk
        // every block. Splitting the whole shard instead would hand one RISC entire leading
        // indices and put the block boundary in the middle of its range.
        const uint32_t sticks_per_block = input_num_sticks_per_block[input_id];
        const auto input_num_sticks_per_risc = tt::div_up(sticks_per_block, 2);
        runtime_args_0.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_0.push_back(input_num_sticks_per_risc);
        runtime_args_0.push_back(input_write_offsets[input_id]);
        runtime_args_0.push_back(0);
        runtime_args_0.push_back(input_block_strides[input_id]);
        runtime_args_1.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_1.push_back(sticks_per_block - input_num_sticks_per_risc);
        runtime_args_1.push_back(input_write_offsets[input_id] + (output_stride * input_num_sticks_per_risc));
        runtime_args_1.push_back(page_size * input_num_pages_per_stick[input_id] * input_num_sticks_per_risc);
        runtime_args_1.push_back(input_block_strides[input_id]);
    }

    // Match the legacy CachedProgram path: SetRuntimeArgs(..., all_cores, args).
    // These values must live in per-core runtime args, not common_runtime_args.
    // BRISC (writer) and NCRISC (reader) each have their own RTA region; using
    // common_runtime_args for both kernels made both RISCs observe the same
    // offsets (precision failures in sharded concat).
    KernelDescriptor::CoreRuntimeArgs reader_rt_args(runtime_args_0.begin(), runtime_args_0.end());
    KernelDescriptor::CoreRuntimeArgs writer_rt_args(runtime_args_1.begin(), runtime_args_1.end());

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.runtime_args.reserve(all_cores.num_cores());
    for (const auto& range : all_cores.ranges()) {
        for (const CoreCoord& core : range) {
            reader_desc.runtime_args.emplace_back(core, reader_rt_args);
        }
    }
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = compile_time_args;
    writer_desc.runtime_args.reserve(all_cores.num_cores());
    for (const auto& range : all_cores.ranges()) {
        for (const CoreCoord& core : range) {
            writer_desc.runtime_args.emplace_back(core, writer_rt_args);
        }
    }
    writer_desc.config = WriterConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim

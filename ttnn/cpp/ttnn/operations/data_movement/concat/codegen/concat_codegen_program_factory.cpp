// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr uint32_t kCbOut = 0;
constexpr uint32_t kCbScratch = 1;

constexpr const char* kConcatKernelDir = "ttnn/cpp/ttnn/operations/data_movement/concat/codegen/kernels/";
constexpr const char* kCommonKernelDir = "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/";

constexpr const char* kReaderRmInterleaved = "reader_concat_rm_interleaved.cpp";
constexpr const char* kReaderRmNonwidthNway = "reader_concat_rm_nonwidth_nway.cpp";
constexpr const char* kReaderRmWidthInterleaved = "reader_concat_rm_width_interleaved.cpp";
constexpr const char* kReaderRmWidthNway = "reader_concat_rm_width_nway.cpp";
constexpr const char* kWriterInterleaved = "writer_interleaved.cpp";

using RtArgs = std::vector<std::variant<uint32_t, Buffer*>>;

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

// row_wise=false for both the split and the core walk so the per-core ranges line up.
CoreSplit split_work(const Tensor& input, uint32_t total_work) {
    const auto grid_size = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return CoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t work_for_core(const CoreSplit& split, const CoreCoord& core) {
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

CBDescriptor make_cb(
    const CoreRangeSet& cores, uint32_t index, tt::DataFormat format, uint32_t page_size, uint32_t num_pages) {
    return CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = page_size,
        }}},
    };
}

// Where a core that starts at output page `start` picks up each input: the tensor it is reading, how far into that
// tensor's current block it is, and the first page of every tensor it will touch.
struct BlockCursor {
    uint32_t curr_tensor = 0;
    uint32_t curr_tensor_id = 0;
    std::vector<uint32_t> page_id;
};

BlockCursor block_cursor(uint32_t start, const std::vector<uint32_t>& pages_per_block, uint32_t output_per_block) {
    BlockCursor cursor;
    cursor.page_id.resize(pages_per_block.size());
    const uint32_t block_id = start / output_per_block;
    uint32_t id_within_block = start % output_per_block;
    for (uint32_t i = 0; i < pages_per_block.size(); ++i) {
        cursor.page_id[i] = block_id * pages_per_block[i];
        if (id_within_block == 0) {
            continue;
        }
        if (id_within_block >= pages_per_block[i]) {
            cursor.page_id[i] += pages_per_block[i];
            id_within_block -= pages_per_block[i];
            cursor.curr_tensor = i + 1;
        } else {
            cursor.page_id[i] += id_within_block;
            cursor.curr_tensor = i;
            cursor.curr_tensor_id = id_within_block;
            id_within_block = 0;
        }
    }
    return cursor;
}

KernelDescriptor make_kernel(
    const char* kernel_dir, const char* basename, const CoreRangeSet& cores, std::vector<uint32_t> compile_time_args) {
    KernelDescriptor desc;
    desc.kernel_source = std::string(kernel_dir) + basename;
    desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    desc.core_ranges = cores;
    desc.compile_time_args = std::move(compile_time_args);
    return desc;
}

}  // namespace

// The output is allocated by now, so the live L1 already excludes it; the program hash keys the same plan. Forced
// codegen only checked the static window, so having no live room here is an error.
ProgramDescriptor ConcatCodegenProgramFactory::create_descriptor(
    const ConcatCodegenParams& operation_attributes,
    const ConcatCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    namespace cc = operations::data_movement::concat_codegen;
    const auto& inputs = tensor_args.input_tensors;
    const Tensor& first = inputs.front();

    const auto batches = cc::plan_batches_live(inputs, operation_attributes, /*output_allocated=*/true);
    if (!batches.has_value()) {
        TT_THROW(
            "ttnn.concat: codegen path needs {} B of L1 per core for its circular buffers but only {} B is free",
            cc::cb_footprint_bytes(inputs, operation_attributes, cc::BatchPlan{}),
            operations::data_movement::get_max_l1_space(first));
    }

    const uint32_t num_inputs = inputs.size();
    const uint32_t dim = operation_attributes.dim;
    const bool width = cc::concats_width(inputs, operation_attributes);
    const bool two_input = cc::has_two_inputs(inputs);
    const tt::DataFormat format = datatype_to_dataformat_converter(first.dtype());
    const cc::CbPages pages = cc::cb_pages(inputs, operation_attributes);

    Buffer* dst_buffer = tensor_return_value.buffer();
    const auto& out_shape = tensor_return_value.logical_shape();
    const uint32_t num_output_pages = out_shape.volume() / out_shape[cc::kLastAxis];

    std::vector<Buffer*> src_buffers;
    std::vector<uint32_t> src_page_bytes;
    std::vector<uint32_t> src_stick_bytes;
    for (const Tensor& input : inputs) {
        src_buffers.push_back(input.buffer());
        src_page_bytes.push_back(input.buffer()->aligned_page_size());
        src_stick_bytes.push_back(input.logical_shape()[cc::kLastAxis] * input.element_size());
    }

    // Non-width concat interleaves whole blocks of sticks from each input, one block per outer prefix.
    std::vector<uint32_t> pages_per_block(num_inputs, 1);
    uint32_t output_per_block = 1;
    if (!width) {
        uint32_t accum_pages = 1;
        const auto rank = first.logical_shape().rank();
        for (uint32_t d = dim + 1; d + 1 < rank; ++d) {
            accum_pages *= first.logical_shape()[d];
        }
        output_per_block = 0;
        for (uint32_t i = 0; i < num_inputs; ++i) {
            pages_per_block[i] = accum_pages * inputs[i].logical_shape()[dim];
            output_per_block += pages_per_block[i];
        }
    }

    const CoreSplit split = split_work(first, num_output_pages);

    ProgramDescriptor desc;
    const uint32_t depth = cc::cb_depth(batches->read_batch, batches->write_batch);
    desc.cbs.push_back(make_cb(split.all_cores, kCbOut, format, pages.out_page, depth));
    if (pages.scratch_page > 0) {
        desc.cbs.push_back(make_cb(split.all_cores, kCbScratch, format, pages.scratch_page, 1));
    }

    std::vector<uint32_t> reader_ct_args;
    const char* reader_source = nullptr;
    if (!width && two_input) {
        reader_source = kReaderRmInterleaved;
        reader_ct_args = {
            kCbOut,
            batches->read_batch,
            pages_per_block[0],
            pages_per_block[1],
            pages.out_page,
            src_page_bytes[0],
            src_page_bytes[1]};
        TensorAccessorArgs(*src_buffers[0]).append_to(reader_ct_args);
        TensorAccessorArgs(*src_buffers[1]).append_to(reader_ct_args);
    } else if (!width) {
        reader_source = kReaderRmNonwidthNway;
        reader_ct_args = {kCbOut, batches->read_batch, num_inputs, pages.out_page};
        reader_ct_args.insert(reader_ct_args.end(), src_page_bytes.begin(), src_page_bytes.end());
        for (const Buffer* src_buffer : src_buffers) {
            TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
        }
    } else if (two_input) {
        reader_source = kReaderRmWidthInterleaved;
        reader_ct_args = {
            kCbOut,
            src_stick_bytes[0],
            src_stick_bytes[1],
            pages.out_page,
            src_page_bytes[0],
            src_page_bytes[1],
            kCbScratch,
            src_buffers[1]->alignment(),
            batches->read_batch};
        TensorAccessorArgs(*src_buffers[0]).append_to(reader_ct_args);
        TensorAccessorArgs(*src_buffers[1]).append_to(reader_ct_args);
    } else {
        reader_source = kReaderRmWidthNway;
        uint32_t src_alignment = 0;
        for (const Buffer* src_buffer : src_buffers) {
            src_alignment = std::max(src_alignment, src_buffer->alignment());
        }
        reader_ct_args = {kCbOut, kCbScratch, num_inputs, pages.out_page, batches->read_batch, src_alignment};
        for (const Buffer* src_buffer : src_buffers) {
            TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
        }
    }

    KernelDescriptor reader_desc =
        make_kernel(kConcatKernelDir, reader_source, split.all_cores, std::move(reader_ct_args));
    reader_desc.config = ReaderConfigDescriptor{};

    // A whole aligned page per write: a sub-burst DRAM write for an unaligned stick costs more than the padding it
    // saves.
    std::vector<uint32_t> writer_ct_args = {kCbOut, dst_buffer->aligned_page_size(), batches->write_batch};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelDescriptor writer_desc =
        make_kernel(kCommonKernelDir, kWriterInterleaved, split.all_cores, std::move(writer_ct_args));
    writer_desc.config = WriterConfigDescriptor{};

    // Source addresses and per-input geometry are identical on every core, so they travel as common args.
    KernelDescriptor::RTArgList reader_common_args;
    for (Buffer* src_buffer : src_buffers) {
        reader_common_args.push_back(src_buffer);
    }
    if (!width && !two_input) {
        for (const uint32_t pages : pages_per_block) {
            reader_common_args.push_back(pages);
        }
    } else if (width && !two_input) {
        for (const uint32_t stick_bytes : src_stick_bytes) {
            reader_common_args.push_back(stick_bytes);
        }
        for (const uint32_t page_bytes : src_page_bytes) {
            reader_common_args.push_back(page_bytes);
        }
    }
    reader_desc.emplace_common_runtime_args(reader_common_args);

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);

        if (!width) {
            const BlockCursor cursor = block_cursor(start, pages_per_block, output_per_block);
            RtArgs args = {n, cursor.curr_tensor, cursor.curr_tensor_id};
            args.insert(args.end(), cursor.page_id.begin(), cursor.page_id.end());
            reader_desc.emplace_runtime_args(core, args);
        } else if (two_input) {
            reader_desc.emplace_runtime_args(core, RtArgs{n, start, start});
        } else {
            reader_desc.emplace_runtime_args(core, RtArgs{n, start});
        }

        writer_desc.emplace_runtime_args(core, RtArgs{dst_buffer, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim

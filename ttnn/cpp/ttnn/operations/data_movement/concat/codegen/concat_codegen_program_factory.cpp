// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

std::optional<ConcatCbPlan> plan_concat_cb(uint32_t page_size, uint32_t max_batch, uint64_t l1_budget_bytes) {
    if (page_size == 0) {
        return ConcatCbPlan{max_batch, 2 * max_batch};
    }
    const uint64_t double_buffered_fit = l1_budget_bytes / (2ull * page_size);
    if (double_buffered_fit > 0) {
        const uint32_t batch = static_cast<uint32_t>(std::min<uint64_t>(max_batch, double_buffered_fit));
        return ConcatCbPlan{batch, 2 * batch};
    }
    // Double buffering doesn't fit even at batch=1; fall back to the single-
    // buffered BATCH<=1 kernel path, which only needs depth=1.
    if (static_cast<uint64_t>(page_size) <= l1_budget_bytes) {
        return ConcatCbPlan{1, 1};
    }
    return std::nullopt;
}

namespace {

// Names here that a sibling codegen factory also uses carry a Concat prefix: a unity build
// merges this anonymous namespace with theirs, and untilize and repeat declare their own
// kKernelDir, kCbIn, CoreSplit and work_for_core in the same enclosing ttnn::prim.

// Host half of the block-cycling cursor the non-width readers share, inlined for two
// inputs and factored out for N. Maps a core's first output stick to the per-input
// reader cursor: inputs before the current one are advanced a full block,
// inputs after remain at the current block, and the current input carries the
// within-block offset.
struct RmNwayCursor {
    uint32_t current_input = 0;
    uint32_t current_input_stick = 0;
    std::vector<uint32_t> stick_ids;
};

RmNwayCursor rm_nway_reader_cursor(const std::vector<uint32_t>& sticks_per_block, uint32_t output_start_stick) {
    uint32_t block_sticks = 0;
    for (uint32_t sticks : sticks_per_block) {
        block_sticks += sticks;
    }
    const uint32_t block_id = output_start_stick / block_sticks;
    uint32_t remaining = output_start_stick % block_sticks;

    RmNwayCursor cursor;
    cursor.stick_ids.resize(sticks_per_block.size());
    for (size_t i = 0; i < sticks_per_block.size(); ++i) {
        cursor.stick_ids[i] = block_id * sticks_per_block[i];
    }
    for (size_t i = 0; i < sticks_per_block.size(); ++i) {
        if (remaining >= sticks_per_block[i]) {
            cursor.stick_ids[i] += sticks_per_block[i];
            remaining -= sticks_per_block[i];
            cursor.current_input = static_cast<uint32_t>(i) + 1;
        } else {
            cursor.stick_ids[i] += remaining;
            cursor.current_input = static_cast<uint32_t>(i);
            cursor.current_input_stick = remaining;
            break;
        }
    }
    return cursor;
}

// How many output sticks correspond to one unit of the concat dimension, for a
// non-width `dim` (dim < ndim - 1).
uint32_t num_accum_sticks(const ttnn::Shape& out_shape, uint32_t dim) {
    const uint32_t ndim = out_shape.rank();
    uint32_t accum = 1;
    for (uint32_t i = dim + 1; i < ndim; ++i) {
        accum *= out_shape[i];
    }
    if (ndim > 1 && dim < ndim - 1) {
        accum /= out_shape[-1];
    }
    return accum;
}

struct ConcatCoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

// split_work_to_cores / corerange_to_cores at their row_wise=False default, which is
// the core ordering the sibling codegen data-movement ports also assume.
ConcatCoreSplit concat_split_work(IDevice* device, uint32_t total_work) {
    const auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return ConcatCoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t concat_work_for_core(const ConcatCoreSplit& split, const CoreCoord& core) {
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

constexpr const char* kConcatKernelDir = "ttnn/cpp/ttnn/operations/data_movement/concat/codegen/kernels/";
constexpr const char* kConcatSharedWriter =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp";
constexpr uint32_t kConcatCbIn = 0;
constexpr uint32_t kConcatCbScratch = 1;

// build_concat_rm: 2-tensor RM, non-width dim. reader_concat_rm_interleaved.cpp.
ProgramDescriptor create_descriptor_rm(
    IDevice* device,
    const std::vector<Tensor>& input_tensors,
    Tensor& output,
    uint32_t dim,
    uint32_t total_out_sticks) {
    const Tensor& in0 = input_tensors[0];
    const Tensor& in1 = input_tensors[1];
    Buffer* src0 = in0.buffer();
    Buffer* src1 = in1.buffer();
    Buffer* dst = output.buffer();

    const uint32_t out_page = static_cast<uint32_t>(dst->aligned_page_size());
    const uint32_t cb_page = std::max(
        {out_page, static_cast<uint32_t>(src0->aligned_page_size()), static_cast<uint32_t>(src1->aligned_page_size())});
    const auto plan = plan_concat_cb(cb_page, kConcatNonWidthBatch, operations::data_movement::get_max_l1_space(in0));
    TT_FATAL(plan.has_value(), "ConcatCodegen: RM concat CB page ({} B) does not fit per-core L1", cb_page);
    const uint32_t batch = plan->batch;

    const uint32_t accum = num_accum_sticks(output.logical_shape(), dim);
    const uint32_t ppb_0 = accum * in0.logical_shape()[dim];
    const uint32_t ppb_1 = accum * in1.logical_shape()[dim];

    const ConcatCoreSplit split = concat_split_work(device, total_out_sticks);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan->depth * cb_page,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kConcatCbIn,
            .data_format = cb_data_format,
            .page_size = cb_page,
        }}},
    });

    std::vector<uint32_t> reader_ct = {
        kConcatCbIn,
        batch,
        2,
        ppb_0,
        ppb_1,
        cb_page,
        static_cast<uint32_t>(src0->aligned_page_size()),
        static_cast<uint32_t>(src1->aligned_page_size())};
    TensorAccessorArgs(*src0).append_to(reader_ct);
    TensorAccessorArgs(*src1).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kConcatKernelDir) + "reader_concat_rm_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {kConcatCbIn, out_page};
    TensorAccessorArgs(*dst).append_to(writer_ct);
    writer_ct.push_back(batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kConcatSharedWriter;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = concat_work_for_core(split, core);
        const RmNwayCursor cursor = rm_nway_reader_cursor({ppb_0, ppb_1}, start);
        reader_desc.emplace_runtime_args(
            core,
            {n,
             cursor.current_input,
             cursor.current_input_stick,
             src0,
             src1,
             cursor.stick_ids[0],
             cursor.stick_ids[1]});
        writer_desc.emplace_runtime_args(core, {dst, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

// build_concat_rm_width: 2-tensor RM, width dim. reader_concat_rm_width_interleaved.cpp.
ProgramDescriptor create_descriptor_rm_width(
    IDevice* device, const std::vector<Tensor>& input_tensors, Tensor& output, uint32_t total_out_sticks) {
    const Tensor& in0 = input_tensors[0];
    const Tensor& in1 = input_tensors[1];
    Buffer* src0 = in0.buffer();
    Buffer* src1 = in1.buffer();
    Buffer* dst = output.buffer();

    const uint32_t in0_stick = static_cast<uint32_t>(src0->page_size());
    const uint32_t in1_stick = static_cast<uint32_t>(src1->page_size());
    const uint32_t in0_page = static_cast<uint32_t>(src0->aligned_page_size());
    const uint32_t in1_page = static_cast<uint32_t>(src1->aligned_page_size());
    const uint32_t out_page = static_cast<uint32_t>(dst->aligned_page_size());
    const uint32_t in1_noc_alignment = src1->alignment();

    // Scratch CB granularity headroom: composing an unaligned second stick directly
    // into the assembly CB corrupts it, seen on an 8+16+16+10 B bf16 cascade.
    const uint32_t l1_cb_granularity = device->allocator()->get_alignment(BufferType::L1);
    const uint32_t scratch_page = std::max(in0_page, in1_page) + l1_cb_granularity;

    const uint64_t l1_budget = operations::data_movement::get_max_l1_space(in0);
    TT_FATAL(scratch_page <= l1_budget, "ConcatCodegen: RM width-concat scratch CB does not fit per-core L1");
    const auto plan = plan_concat_cb(out_page, kConcatWidthWriteBatch, l1_budget - scratch_page);
    TT_FATAL(plan.has_value(), "ConcatCodegen: RM width-concat CB page ({} B) does not fit per-core L1", out_page);
    const uint32_t write_batch = plan->batch;

    const ConcatCoreSplit split = concat_split_work(device, total_out_sticks);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan->depth * out_page,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kConcatCbIn, .data_format = cb_data_format, .page_size = out_page}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch_page,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kConcatCbScratch, .data_format = cb_data_format, .page_size = scratch_page}}},
    });

    // Read batch matches write_batch: the CB depth above is only actually
    // pipelined if the reader fills it write_batch pages at a time instead of
    // reserving/barriering one page per stick.
    std::vector<uint32_t> reader_ct = {
        kConcatCbIn,
        in0_stick,
        in1_stick,
        out_page,
        in0_page,
        in1_page,
        kConcatCbScratch,
        in1_noc_alignment,
        write_batch};
    TensorAccessorArgs(*src0).append_to(reader_ct);
    TensorAccessorArgs(*src1).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kConcatKernelDir) + "reader_concat_rm_width_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {kConcatCbIn, out_page};
    TensorAccessorArgs(*dst).append_to(writer_ct);
    writer_ct.push_back(write_batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kConcatSharedWriter;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = concat_work_for_core(split, core);
        reader_desc.emplace_runtime_args(core, {n, src0, src1, start, start});
        writer_desc.emplace_runtime_args(core, {dst, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

// build_concat_rm_nonwidth_nway: N>2 RM, non-width dim. reader_concat_rm_nonwidth_nway.cpp.
ProgramDescriptor create_descriptor_rm_nonwidth_nway(
    IDevice* device,
    const std::vector<Tensor>& input_tensors,
    Tensor& output,
    uint32_t dim,
    uint32_t total_out_sticks) {
    const uint32_t n_inputs = static_cast<uint32_t>(input_tensors.size());
    const Tensor& in0 = input_tensors[0];
    Buffer* src0 = in0.buffer();
    Buffer* dst = output.buffer();

    const uint32_t in_page = static_cast<uint32_t>(src0->aligned_page_size());
    const uint32_t out_page = static_cast<uint32_t>(dst->aligned_page_size());
    const uint32_t cb_page = std::max(in_page, out_page);
    const auto plan = plan_concat_cb(cb_page, kConcatNonWidthBatch, operations::data_movement::get_max_l1_space(in0));
    TT_FATAL(plan.has_value(), "ConcatCodegen: RM N-way concat CB page ({} B) does not fit per-core L1", cb_page);
    const uint32_t batch = plan->batch;

    const uint32_t accum = num_accum_sticks(output.logical_shape(), dim);
    std::vector<uint32_t> sticks_per_block(n_inputs);
    for (uint32_t i = 0; i < n_inputs; ++i) {
        sticks_per_block[i] = accum * input_tensors[i].logical_shape()[dim];
    }

    const ConcatCoreSplit split = concat_split_work(device, total_out_sticks);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan->depth * cb_page,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kConcatCbIn, .data_format = cb_data_format, .page_size = cb_page}}},
    });

    std::vector<uint32_t> reader_ct = {kConcatCbIn, batch, n_inputs, cb_page, in_page};
    TensorAccessorArgs(*src0).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kConcatKernelDir) + "reader_concat_rm_nonwidth_nway.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {kConcatCbIn, out_page};
    TensorAccessorArgs(*dst).append_to(writer_ct);
    writer_ct.push_back(batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kConcatSharedWriter;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = concat_work_for_core(split, core);
        const RmNwayCursor cursor = rm_nway_reader_cursor(sticks_per_block, start);

        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(3 + 3 * n_inputs);
        reader_args.push_back(n);
        reader_args.push_back(cursor.current_input);
        reader_args.push_back(cursor.current_input_stick);
        for (uint32_t i = 0; i < n_inputs; ++i) {
            reader_args.push_back(input_tensors[i].buffer());
        }
        reader_args.append(sticks_per_block);
        reader_args.append(cursor.stick_ids);

        reader_desc.emplace_runtime_args(core, reader_args);
        writer_desc.emplace_runtime_args(core, {dst, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

// build_concat_rm_width_nway: N>2 RM, width dim. reader_concat_rm_width_nway.cpp.
ProgramDescriptor create_descriptor_rm_width_nway(
    IDevice* device, const std::vector<Tensor>& input_tensors, Tensor& output, uint32_t total_out_sticks) {
    const uint32_t n_inputs = static_cast<uint32_t>(input_tensors.size());
    Buffer* dst = output.buffer();
    const uint32_t out_page = static_cast<uint32_t>(dst->aligned_page_size());

    std::vector<uint32_t> stick_sizes(n_inputs);
    std::vector<uint32_t> page_sizes(n_inputs);
    uint32_t scratch_page = 0;
    for (uint32_t i = 0; i < n_inputs; ++i) {
        Buffer* buf = input_tensors[i].buffer();
        stick_sizes[i] = static_cast<uint32_t>(buf->page_size());
        page_sizes[i] = static_cast<uint32_t>(buf->aligned_page_size());
        scratch_page = std::max(scratch_page, page_sizes[i]);
    }

    const uint64_t l1_budget = operations::data_movement::get_max_l1_space(input_tensors[0]);
    TT_FATAL(scratch_page <= l1_budget, "ConcatCodegen: RM N-way width-concat scratch CB does not fit per-core L1");
    const auto plan = plan_concat_cb(out_page, kConcatWidthWriteBatch, l1_budget - scratch_page);
    TT_FATAL(
        plan.has_value(), "ConcatCodegen: RM N-way width-concat CB page ({} B) does not fit per-core L1", out_page);
    const uint32_t write_batch = plan->batch;
    const ConcatCoreSplit split = concat_split_work(device, total_out_sticks);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan->depth * out_page,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kConcatCbIn, .data_format = cb_data_format, .page_size = out_page}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch_page,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kConcatCbScratch, .data_format = cb_data_format, .page_size = scratch_page}}},
    });

    // All N inputs share one memory configuration (enforced by
    // supported_by_codegen()), so one buffer's transport alignment answers
    // for every input's direct-write destination-offset check.
    const uint32_t noc_alignment = static_cast<uint32_t>(input_tensors[0].buffer()->alignment());
    // Read batch matches write_batch, same as the 2-tensor width builder: the
    // CB depth above is only pipelined if the reader fills it write_batch
    // pages at a time instead of one page per barrier.
    std::vector<uint32_t> reader_ct = {kConcatCbIn, kConcatCbScratch, n_inputs, out_page, write_batch, noc_alignment};
    TensorAccessorArgs(*input_tensors[0].buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kConcatKernelDir) + "reader_concat_rm_width_nway.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {kConcatCbIn, out_page};
    TensorAccessorArgs(*dst).append_to(writer_ct);
    writer_ct.push_back(write_batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kConcatSharedWriter;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = concat_work_for_core(split, core);

        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(2 + 3 * n_inputs);
        reader_args.push_back(n);
        reader_args.push_back(start);
        for (uint32_t i = 0; i < n_inputs; ++i) {
            reader_args.push_back(input_tensors[i].buffer());
        }
        reader_args.append(stick_sizes);
        reader_args.append(page_sizes);

        reader_desc.emplace_runtime_args(core, reader_args);
        writer_desc.emplace_runtime_args(core, {dst, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace

ProgramDescriptor ConcatCodegenProgramFactory::create_descriptor(
    const ConcatCodegenParams& operation_attributes,
    const ConcatCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    IDevice* device = output.device();
    const uint32_t ndim = output.logical_shape().rank();
    const uint32_t dim = operation_attributes.dim;
    const bool is_width = (dim == ndim - 1);
    const uint32_t total_out_sticks = operation_attributes.total_out_sticks;

    if (input_tensors.size() == 2) {
        return is_width ? create_descriptor_rm_width(device, input_tensors, output, total_out_sticks)
                        : create_descriptor_rm(device, input_tensors, output, dim, total_out_sticks);
    }
    return is_width ? create_descriptor_rm_width_nway(device, input_tensors, output, total_out_sticks)
                    : create_descriptor_rm_nonwidth_nway(device, input_tensors, output, dim, total_out_sticks);
}

}  // namespace ttnn::prim

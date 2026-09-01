// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Pages per read / write batch, shared by the TILE and RM branches below. Prefixed: unity builds
// merge anonymous namespaces across TUs, so unprefixed names collide with
// repeat_codegen_program_factory.cpp.
constexpr uint32_t kRiReadBatch = 4;
constexpr uint32_t kRiWriteBatch = 4;
// Double-buffers whichever side batches more, so one batch fills while the other drains.
constexpr uint32_t kRiCbDepth = 2 * std::max(kRiReadBatch, kRiWriteBatch);

// SEQ_REPEAT_INTERLEAVE in common/kernels/codegen/sequencers.h, which the unified tile reader
// switches on at compile time.
constexpr uint32_t kSeqRepeatInterleave = 9;

// Shared by several data_movement ops, hence common/ rather than this op's own kernels dir.
constexpr const char* kTileReaderSrc =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp";
constexpr const char* kWriterSrc =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp";
constexpr const char* kRmReaderSrc =
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/"
    "reader_repeat_interleave_rm.cpp";

struct CoreWork {
    CoreCoord core;
    uint32_t num_pages;
    uint32_t start_page;
};

std::vector<CoreWork> layout_cores(
    const std::vector<CoreCoord>& cores,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t units_per_core_group_1,
    uint32_t units_per_core_group_2) {
    std::vector<CoreWork> layout;
    layout.reserve(cores.size());
    uint32_t start_page = 0;
    for (const auto& core : cores) {
        uint32_t num_pages;
        if (core_group_1.contains(core)) {
            num_pages = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages = units_per_core_group_2;
        } else {
            TT_THROW("repeat_interleave codegen: core not in either work-split group");
        }
        layout.push_back({core, num_pages, start_page});
        start_page += num_pages;
    }
    return layout;
}

}  // namespace

ProgramDescriptor RepeatInterleaveCodegenProgramFactory::create_descriptor(
    const RepeatInterleaveCodegenParams& operation_attributes,
    const RepeatInterleaveCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* in_buffer = input.buffer();
    Buffer* out_buffer = output.buffer();

    auto grid = input.device()->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, operation_attributes.total_out_pages);
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    const auto layout = layout_cores(cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2);

    ProgramDescriptor desc;

    if (input.layout() == Layout::TILE) {
        // TILE outer-dim path: unified reader on the REPEAT_INTERLEAVE sequencer + the shared
        // interleaved writer. The CB slot is one tile whatever the shape, so unlike the RM branch
        // below its footprint needs no L1 bound.
        constexpr uint32_t cb_id = CBIndex::c_0;
        const auto out_data_format = datatype_to_dataformat_converter(output.dtype());
        const uint32_t cb_page_size = tile_size(out_data_format);

        desc.cbs.push_back(CBDescriptor{
            .total_size = kRiCbDepth * cb_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_id,
                .data_format = out_data_format,
                .page_size = cb_page_size,
            }}},
        });

        KernelDescriptor reader_desc;
        reader_desc.kernel_source = kTileReaderSrc;
        reader_desc.core_ranges = all_cores;
        TensorAccessorArgs(*in_buffer).append_to(reader_desc.compile_time_args);
        // reader_tile_interleaved_unified.cpp reads "src_page_pitch" unconditionally for every
        // seq_id (not just the ones that override it); 0 keeps the accessor's own page size.
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqRepeatInterleave}, {"cb_id", cb_id}, {"batch", kRiReadBatch}, {"src_page_pitch", 0}};
        reader_desc.config = ReaderConfigDescriptor{};

        const uint32_t out_page_size = static_cast<uint32_t>(out_buffer->aligned_page_size());
        KernelDescriptor writer_desc;
        writer_desc.kernel_source = kWriterSrc;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = {cb_id, out_page_size};
        TensorAccessorArgs(*out_buffer).append_to(writer_desc.compile_time_args);
        writer_desc.compile_time_args.push_back(kRiWriteBatch);
        writer_desc.config = WriterConfigDescriptor{};

        for (const auto& work : layout) {
            reader_desc.emplace_runtime_args(
                work.core,
                {in_buffer,
                 work.num_pages,
                 work.start_page,
                 operation_attributes.num_repeats,
                 operation_attributes.lower_pages,
                 operation_attributes.rep_dim_pages});
            writer_desc.emplace_runtime_args(work.core, {out_buffer, work.num_pages, work.start_page});
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        return desc;
    }

    // ROW_MAJOR whole-stick (outer/H) path. The within-stick (last-dim) case has no reader wired
    // here; validate_on_program_cache_miss rejects it before this factory runs.
    //
    // An RM slot is a whole stick, so kRiReadBatch/kRiWriteBatch worth of them is not guaranteed to
    // fit: shrink the batch (and with it the CB depth) to what per-core L1 admits. Validation
    // rejects anything below kRmCbMinSlots, so the loop always terminates with batch >= 1.
    const auto cb_budget =
        ttnn::operations::data_movement::repeat_interleave_codegen::rm_cb_budget(input, output.memory_config());
    const uint32_t slot_stride = cb_budget.slot_stride;
    uint32_t batch = kRiReadBatch;
    while (2 * batch > cb_budget.max_slots) {
        batch /= 2;
    }
    const uint32_t cb_depth = 2 * batch;

    constexpr uint32_t cb_id = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_depth * slot_stride,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = slot_stride,
        }}},
    });

    const uint32_t in_page_size = static_cast<uint32_t>(in_buffer->aligned_page_size());
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kRmReaderSrc;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = {operation_attributes.stick_size, in_page_size, slot_stride};
    TensorAccessorArgs(*in_buffer).append_to(reader_desc.compile_time_args);
    reader_desc.compile_time_args.push_back(cb_id);
    reader_desc.compile_time_args.push_back(operation_attributes.num_repeats);
    reader_desc.compile_time_args.push_back(operation_attributes.lower_pages);
    reader_desc.compile_time_args.push_back(operation_attributes.rep_dim_pages);
    reader_desc.compile_time_args.push_back(batch);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterSrc;
    writer_desc.core_ranges = all_cores;
    // CT[1] is the requested transfer size; the writer takes the destination pitch from the accessor
    // and the L1 slot stride from the CB descriptor, then clamps the transfer to the smallest of the
    // three. Requesting the stick keeps each output page's alignment padding untouched, which is what
    // the host expects to trim on read-back.
    writer_desc.compile_time_args = {cb_id, operation_attributes.stick_size};
    TensorAccessorArgs(*out_buffer).append_to(writer_desc.compile_time_args);
    writer_desc.compile_time_args.push_back(batch);
    writer_desc.config = WriterConfigDescriptor{};

    for (const auto& work : layout) {
        reader_desc.emplace_runtime_args(work.core, {in_buffer, work.num_pages, work.start_page});
        writer_desc.emplace_runtime_args(work.core, {out_buffer, work.num_pages, work.start_page});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim

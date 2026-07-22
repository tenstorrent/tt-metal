// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// ops/repeat/spec.py: READ_BATCH / WRITE_BATCH / _CB_DEPTH (shared verbatim by repeat_interleave's
// TILE and RM builders).
constexpr uint32_t kReadBatch = 4;
constexpr uint32_t kWriteBatch = 4;
constexpr uint32_t kCbDepth = std::max(2 * std::max(kReadBatch, kWriteBatch), 8u);

// sequencers.h SEQ_REPEAT_INTERLEAVE (ops/repeat_interleave/builder.py's SEQ_REPEAT_INTERLEAVE).
constexpr uint32_t kSeqRepeatInterleave = 9;

// repeat_interleave_codegen_device_operation.cpp's rep_dim left-padding rank.
constexpr uint32_t kRepDimPadRank = 4;

constexpr const char* kTileReaderSrc =
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/"
    "reader_tile_interleaved_unified.cpp";
constexpr const char* kTileWriterSrc =
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/writer_interleaved.cpp";
constexpr const char* kRmReaderSrc =
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/"
    "reader_repeat_interleave_rm.cpp";
constexpr const char* kRmWriterSrc =
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/"
    "writer_repeat_interleave_rm.cpp";

uint32_t recover_rep_dim(uint32_t padded_rep_dim, uint32_t ndim) { return padded_rep_dim - (kRepDimPadRank - ndim); }

uint32_t align_up(uint32_t value, uint32_t alignment) { return ((value + alignment - 1) / alignment) * alignment; }

// ops/repeat_interleave/builder.py::_page_alignment: L1 interleaved -> L1 alignment; DRAM -> the
// architecture's DRAM alignment. Queried from the real device HAL, never a sweep constant.
uint32_t page_alignment(const MemoryConfig& memory_config) {
    return memory_config.buffer_type() == BufferType::L1 ? hal::get_l1_alignment() : hal::get_dram_alignment();
}

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

    const uint32_t ndim = input.logical_shape().rank();
    const uint32_t rep_dim = recover_rep_dim(operation_attributes.rep_dim, ndim);

    auto grid = input.device()->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, operation_attributes.total_out_pages);
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    const auto layout = layout_cores(cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2);

    ProgramDescriptor desc;

    if (input.layout() == Layout::TILE) {
        // TILE outer-dim path: unified seq_id=9 (REPEAT_INTERLEAVE) reader + shared interleaved
        // writer (ops/repeat_interleave/spec.py::build_repeat_tile, via ops/repeat/spec.py's
        // shared build_repeat_tile / _plan / _writer_ct).
        constexpr uint32_t cb_id = CBIndex::c_0;
        const auto out_data_format = datatype_to_dataformat_converter(output.dtype());
        const uint32_t cb_page_size = tile_size(out_data_format);

        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbDepth * cb_page_size,
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
            {"seq_id", kSeqRepeatInterleave}, {"cb_id", cb_id}, {"batch", kReadBatch}, {"src_page_pitch", 0}};
        reader_desc.config = ReaderConfigDescriptor{};

        const uint32_t out_page_size = align_up(cb_page_size, page_alignment(output.memory_config()));
        KernelDescriptor writer_desc;
        writer_desc.kernel_source = kTileWriterSrc;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = {cb_id, out_page_size};
        TensorAccessorArgs(*out_buffer).append_to(writer_desc.compile_time_args);
        writer_desc.compile_time_args.push_back(kWriteBatch);
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

    // ROW_MAJOR whole-stick (outer/H) path (ops/repeat_interleave/spec.py::
    // build_repeat_interleave_rm_factory). supported_by_codegen() unconditionally rejects the
    // within-stick (last-dim) case -- validate_on_program_cache_miss TT_FATALs before this
    // factory ever runs -- so the last-dim kernel (reader_repeat_interleave_lastdim_rm.cpp) is
    // deliberately not wired here; it would be unreachable dead code.
    TT_FATAL(
        rep_dim != ndim - 1,
        "repeat_interleave codegen: RM within-stick (last-dim) replication is not wired in this "
        "program factory; supported_by_codegen() must reject it before create_descriptor runs");

    const uint32_t in_align = page_alignment(input.memory_config());
    const uint32_t out_align = page_alignment(output.memory_config());
    const uint32_t slot_stride = std::max(
        align_up(operation_attributes.stick_size, in_align), align_up(operation_attributes.stick_size, out_align));

    constexpr uint32_t cb_id = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = kCbDepth * slot_stride,
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
    reader_desc.compile_time_args.push_back(kReadBatch);
    reader_desc.config = ReaderConfigDescriptor{};

    const uint32_t out_page_size = static_cast<uint32_t>(out_buffer->aligned_page_size());
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kRmWriterSrc;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {cb_id, operation_attributes.stick_size, out_page_size, slot_stride};
    TensorAccessorArgs(*out_buffer).append_to(writer_desc.compile_time_args);
    writer_desc.compile_time_args.push_back(kWriteBatch);
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

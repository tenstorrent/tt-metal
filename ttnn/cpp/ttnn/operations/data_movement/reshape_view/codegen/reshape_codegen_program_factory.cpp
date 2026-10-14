// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_program_factory.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace rcg = ttnn::operations::data_movement::reshape_codegen;

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/reshape_view/codegen/kernels";
constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kCbMid = tt::CBIndex::c_1;
// Sticks the non-aligned reader gathers per NOC barrier.
constexpr uint32_t kReaderBatchSticks = 8;
constexpr uint32_t kCbPlan = tt::CBIndex::c_0;
constexpr uint32_t kCbStaged = tt::CBIndex::c_1;
constexpr uint32_t kCbRing = tt::CBIndex::c_2;
constexpr uint32_t kCbZero = tt::CBIndex::c_3;
// Datums per face row; a tile-row run never crosses a face boundary, so the writer copies face rows.
constexpr uint32_t kFaceWidth = 16;

std::string op_kernel(const char* name) { return std::string(kKernelDir) + "/" + name; }

CBDescriptor make_cb(
    uint32_t cb_id, tt::DataFormat fmt, uint32_t page_size, uint32_t depth, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = page_size * depth,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = fmt,
            .page_size = page_size,
        }}},
    };
}

struct RmCoreWork {
    uint32_t num_old_reads = 0;
    uint32_t old_per_read = 0;
    uint32_t old_per_push = 0;
    uint32_t num_new_reads = 0;
    uint32_t new_per_read = 0;
    uint32_t new_per_push = 0;
    uint32_t curr_old = 0;
    uint32_t curr_new = 0;
};

RmCoreWork rm_core_work(const rcg::RmPlan& p, uint32_t start, uint32_t n_split) {
    RmCoreWork w;
    const auto reads_for = [&](uint32_t n, uint32_t stick_bytes) {
        return p.push_sticks != 0 ? n / p.push_sticks : merge_num_sticks_to_read(n, stick_bytes, rcg::kMaxReadSize);
    };
    uint32_t n_old = 0;
    uint32_t n_new = 0;
    if (p.split_by_old) {
        n_old = n_split;
        n_new = n_old * p.ratio;
        w.curr_old = start;
        w.curr_new = start * p.ratio;
    } else {
        n_new = n_split;
        n_old = n_new * p.ratio;
        w.curr_new = start;
        w.curr_old = start * p.ratio;
    }

    if (!p.can_coalesce) {
        w.new_per_read = 1;
        w.new_per_push = 1;
        w.num_new_reads = n_new;
        if (p.split_by_old) {
            w.num_old_reads = n_old;
            w.old_per_read = 1;
            w.old_per_push = p.ratio;
        } else {
            w.num_old_reads = n_new;
            w.old_per_read = p.ratio;
            w.old_per_push = 1;
        }
    } else if (p.split_by_old) {
        if (n_old != 0) {
            w.num_old_reads = reads_for(n_old, p.old_stick);
            w.old_per_read = n_old / w.num_old_reads;
            w.old_per_push = w.old_per_read * p.ratio;
            w.new_per_push = w.old_per_push;
            w.new_per_read = w.old_per_push;
            w.num_new_reads = n_new / w.new_per_read;
        }
    } else if (n_new != 0) {
        w.num_new_reads = reads_for(n_new, p.new_stick);
        w.new_per_read = n_new / w.num_new_reads;
        w.new_per_push = w.new_per_read;
        w.old_per_push = w.new_per_push;
        w.old_per_read = w.old_per_push * p.ratio;
        w.num_old_reads = n_old / w.old_per_read;
    }
    return w;
}

ProgramDescriptor build_rm(const rcg::RmPlan& p, const Tensor& input, Tensor& output, const rcg::WorkSplit& split) {
    ProgramDescriptor desc;
    const CoreRangeSet& all_cores = split.all_cores;
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    Buffer* in_buf = input.buffer();
    Buffer* out_buf = output.buffer();

    desc.cbs.push_back(make_cb(kCbIn, fmt, p.cb_page, p.cb_depth, all_cores));
    if (!p.can_coalesce) {
        desc.cbs.push_back(make_cb(kCbMid, fmt, p.scratch_bytes, 1, all_cores));
    }

    KernelDescriptor reader;
    reader.kernel_source = op_kernel("reader_reshape_rm.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = {kCbIn, p.old_stick, p.old_aligned, p.can_coalesce ? 0u : 1u, kReaderBatchSticks};
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = op_kernel("writer_reshape_rm.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = {kCbIn, p.new_stick, p.new_aligned};
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.compile_time_args.push_back(tt::tt_metal::hal::get_noc_max_burst_size_bytes());
    writer.config = WriterConfigDescriptor{};

    const auto cores = corerange_to_cores(split.all_cores, split.num_cores, false);
    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t n = split.group1.contains(core) ? split.units_group1 : split.units_group2;
        const RmCoreWork w = rm_core_work(p, start, n);
        if (p.can_coalesce) {
            reader.emplace_runtime_args(core, {in_buf, w.num_old_reads, w.old_per_read, w.old_per_push, w.curr_old});
        } else {
            reader.emplace_runtime_args(
                core,
                {in_buf,
                 w.num_old_reads,
                 w.old_per_read,
                 w.curr_old,
                 p.old_stick,
                 p.new_stick,
                 p.ratio,
                 p.split_nonaligned ? 1u : 0u,
                 1u});
        }
        writer.emplace_runtime_args(core, {out_buf, w.num_new_reads, w.new_per_read, w.new_per_push, w.curr_new});
        start += n;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

ProgramDescriptor build_rm_arbitrary(
    const rcg::RmArbitraryPlan& p, const Tensor& input, Tensor& output, const rcg::WorkSplit& split) {
    ProgramDescriptor desc;
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    Buffer* in_buf = input.buffer();
    Buffer* out_buf = output.buffer();
    const CoreRangeSet& all_cores = split.all_cores;

    desc.cbs.push_back(make_cb(kCbIn, fmt, p.slot, 2 * p.batch, all_cores));
    desc.cbs.push_back(make_cb(kCbMid, fmt, p.scratch_bytes, 1, all_cores));

    KernelDescriptor reader;
    reader.kernel_source = op_kernel("reader_reshape_rm_arbitrary.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = {
        kCbIn,
        kCbMid,
        p.old_stick,
        p.old_page,
        p.new_stick,
        p.slab,
        p.slot,
        p.slabs_per_stick,
        p.in_alignment,
        tt::tt_metal::hal::get_noc_max_burst_size_bytes(),
        p.batch,
        p.region_stride};
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = op_kernel("writer_reshape_rm_arbitrary.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = {kCbIn, p.new_stick, p.new_page, p.slab, p.slot, p.slabs_per_stick, p.batch};
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};

    const auto cores = corerange_to_cores(all_cores, split.num_cores, false);
    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t n = split.group1.contains(core) ? split.units_group1 : split.units_group2;
        reader.emplace_runtime_args(core, {in_buf, start, n});
        writer.emplace_runtime_args(core, {out_buf, start, n});
        start += n;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

ProgramDescriptor build_tile_rows(
    const rcg::TileRowsLayout& layout,
    uint32_t staged_batches,
    uint32_t tile_bytes,
    const Tensor& input,
    Buffer* plan_buf,
    Tensor& output,
    const CoreRangeSet& grid) {
    ProgramDescriptor desc;
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    Buffer* in_buf = input.buffer();
    Buffer* out_buf = output.buffer();
    const auto [num_cores, all_cores, group1, group2, batches1, batches2] =
        split_work_to_cores(grid, layout.num_batches, false);

    desc.cbs.push_back(make_cb(kCbPlan, tt::DataFormat::UInt32, layout.page_bytes(), 2, all_cores));
    desc.cbs.push_back(make_cb(kCbStaged, fmt, tile_bytes, staged_batches * layout.in_slots, all_cores));
    desc.cbs.push_back(make_cb(kCbRing, fmt, tile_bytes, rcg::kTileRowsWriteGroup, all_cores));
    desc.cbs.push_back(make_cb(kCbZero, fmt, tile_bytes, 1, all_cores));

    KernelDescriptor reader;
    reader.kernel_source = op_kernel("reader_reshape_tile_rows.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = {
        layout.page_bytes(), tile_bytes, kCbPlan, kCbStaged, layout.in_slots, rcg::kTileRowsHeaderWords};
    TensorAccessorArgs(*plan_buf).append_to(reader.compile_time_args);
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = op_kernel("writer_reshape_tile_rows.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = {
        tile_bytes,
        kFaceWidth * static_cast<uint32_t>(input.element_size()),
        kCbPlan,
        kCbStaged,
        kCbRing,
        kCbZero,
        layout.in_slots,
        layout.entry_words(),
        rcg::kTileRowsWriteGroup,
        rcg::kTileRowsHeaderWords,
        rcg::kTileRowsEntryHeaderWords,
        rcg::kZeroSlot,
        rcg::kRunSlotShift,
        rcg::kRunSrcRowShift,
        rcg::kRunDstRowShift,
        rcg::kRunRowMask};
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};

    const auto cores = corerange_to_cores(all_cores, num_cores, false);
    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t end = start + (group1.contains(core) ? batches1 : batches2);
        reader.emplace_runtime_args(core, {in_buf, plan_buf, start, end});
        writer.emplace_runtime_args(core, {out_buf, start, end});
        start = end;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

// ~Tensor force-frees its device buffer, so the table is held through a shared Tensor the workload owns. It lives
// in interleaved DRAM, one table row per page.
std::shared_ptr<Tensor> upload_table(
    std::vector<uint32_t> table, uint32_t rows, uint32_t row_words, const Tensor& input) {
    const TensorSpec spec(
        ttnn::Shape{rows, row_words}, TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return std::make_shared<Tensor>(Tensor::from_vector(std::move(table), spec).to_device(input.device()));
}

WorkloadDescriptor table_workload(
    const std::shared_ptr<Tensor>& table_owner,
    ProgramDescriptor desc,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor workload_descriptor;
    workload_descriptor.buffers.push_back({table_owner, table_owner->buffer()});
    const auto& ranges = tensor_coords.ranges();
    workload_descriptor.programs.reserve(ranges.size());
    for (size_t i = 0; i + 1 < ranges.size(); ++i) {
        workload_descriptor.programs.push_back({ranges[i], desc});
    }
    if (!ranges.empty()) {
        workload_descriptor.programs.push_back({ranges.back(), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace

tt::tt_metal::ProgramDescriptor ReshapeCodegenProgramFactory::create_descriptor(
    const ReshapeCodegenParams& operation_attributes,
    const ReshapeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    const auto& plan = operation_attributes.plan;
    const uint32_t out_aligned_page = static_cast<uint32_t>(tensor_return_value.buffer()->aligned_page_size());
    TT_FATAL(
        out_aligned_page == plan.out_aligned_page,
        "Reshape output page {} differs from the page the plan assumed",
        out_aligned_page);
    const auto grid = rcg::reshape_core_grid(input, operation_attributes.sub_core_grids);

    const auto chosen = rcg::live_rm_choice(
        input,
        operation_attributes.logical_output_shape,
        operation_attributes.output_mem_config.buffer_type(),
        grid.num_cores(),
        plan);
    TT_FATAL(
        chosen.has_value(),
        "ReshapeCodegen: no row-major program fits the {} B of L1 free per core below the lowest L1 allocation; free "
        "L1 tensors or use the native reshape",
        ttnn::operations::data_movement::get_max_l1_space(input));
    if (const auto* stick = std::get_if<rcg::RmPlan>(&*chosen)) {
        return build_rm(*stick, input, tensor_return_value, rcg::split_units(grid, stick->split.units));
    }
    const auto& slab = std::get<rcg::RmArbitraryPlan>(*chosen);
    return build_rm_arbitrary(slab, input, tensor_return_value, rcg::split_units(grid, slab.split.units));
}

tt::tt_metal::WorkloadDescriptor ReshapeCodegenTileRowsProgramFactory::create_workload_descriptor(
    const ReshapeCodegenParams& operation_attributes,
    const ReshapeCodegenInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const Tensor& input = tensor_args.input;
    const auto& in_shape = input.logical_shape();
    const auto& out_shape = operation_attributes.logical_output_shape;
    const auto grid = rcg::reshape_core_grid(input, operation_attributes.sub_core_grids);
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(datatype_to_dataformat_converter(input.dtype()));

    // Routing already sent calls with no fitting plan to native, so only the forced entry reaches this error.
    const auto chosen = rcg::live_tile_rows_choice(input, out_shape, operation_attributes.plan.tile_rows);
    TT_FATAL(
        chosen.has_value(),
        "ReshapeCodegen: no tile-row plan fits the {} B of L1 free per core below the lowest L1 allocation; free L1 "
        "tensors or use the native reshape",
        ttnn::operations::data_movement::get_max_l1_space(input));
    auto plan = rcg::build_tile_rows_plan(in_shape, out_shape, *chosen);
    const rcg::TileRowsLayout layout = plan.layout;
    TT_FATAL(
        tensor_return_value.buffer()->num_pages() == layout.num_output_pages,
        "Reshape output has {} tiles, the tile-row plan expects {}",
        tensor_return_value.buffer()->num_pages(),
        layout.num_output_pages);

    const auto plan_owner = upload_table(std::move(plan.table), layout.num_batches, layout.page_words, input);
    return table_workload(
        plan_owner,
        build_tile_rows(
            layout, chosen->staged_batches, tile_bytes, input, plan_owner->buffer(), tensor_return_value, grid),
        tensor_coords);
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_program_factory.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_mapped_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace rcg = ttnn::operations::data_movement::reshape_codegen;

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/reshape_view/codegen/kernels";
constexpr uint32_t kCbPlan = tt::CBIndex::c_0;
constexpr uint32_t kCbStaged = tt::CBIndex::c_1;
constexpr uint32_t kCbRing = tt::CBIndex::c_2;
constexpr uint32_t kCbZero = tt::CBIndex::c_3;
constexpr uint32_t kCbMap = tt::CBIndex::c_0;
constexpr uint32_t kCbMappedInput = tt::CBIndex::c_1;
constexpr uint32_t kCbScratch = tt::CBIndex::c_2;

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
    TensorAccessorArgs(*plan_buf).append_to(reader.compile_time_args);
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.named_compile_time_args = {
        {"plan_page_bytes", layout.page_bytes()},
        {"tile_bytes", tile_bytes},
        {"plan_cb", kCbPlan},
        {"input_cb", kCbStaged},
        {"in_slots", layout.in_slots},
        {"header_words", rcg::kTileRowsHeaderWords},
    };
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = op_kernel("writer_reshape_tile_rows.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.named_compile_time_args = {
        {"tile_bytes", tile_bytes},
        {"face_row_bytes", tt::constants::FACE_WIDTH * static_cast<uint32_t>(input.element_size())},
        {"plan_cb", kCbPlan},
        {"input_cb", kCbStaged},
        {"ring_cb", kCbRing},
        {"zero_cb", kCbZero},
        {"in_slots", layout.in_slots},
        {"entry_words", layout.entry_words()},
        {"write_group", rcg::kTileRowsWriteGroup},
        {"header_words", rcg::kTileRowsHeaderWords},
        {"entry_header_words", rcg::kTileRowsEntryHeaderWords},
        {"zero_slot", rcg::kZeroSlot},
        {"slot_shift", rcg::kRunSlotShift},
        {"src_row_shift", rcg::kRunSrcRowShift},
        {"dst_row_shift", rcg::kRunDstRowShift},
        {"row_mask", rcg::kRunRowMask},
    };
    writer.config = WriterConfigDescriptor{};

    reader.emplace_common_runtime_args({in_buf, plan_buf});
    writer.emplace_common_runtime_args({out_buf});
    const auto cores = corerange_to_cores(all_cores, num_cores, false);
    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t end = start + (group1.contains(core) ? batches1 : batches2);
        reader.emplace_runtime_args(core, {start, end});
        writer.emplace_runtime_args(core, {start, end});
        start = end;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

ProgramDescriptor build_tile_mapped(
    const rcg::TileMappedLayout& layout,
    uint32_t input_depth,
    uint32_t tile_bytes,
    const Tensor& input,
    Buffer* map_buf,
    Tensor& output,
    const CoreRangeSet& grid) {
    ProgramDescriptor desc;
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    Buffer* in_buf = input.buffer();
    Buffer* out_buf = output.buffer();
    const auto [num_cores, all_cores, group1, group2, rows1, rows2] = split_work_to_cores(grid, layout.num_rows, false);

    desc.cbs.push_back(make_cb(kCbMap, tt::DataFormat::UInt32, layout.page_bytes(), 1, all_cores));
    desc.cbs.push_back(make_cb(kCbMappedInput, fmt, tile_bytes, input_depth, all_cores));
    desc.cbs.push_back(make_cb(kCbScratch, fmt, tile_bytes, 1, all_cores));

    KernelDescriptor reader;
    reader.kernel_source = op_kernel("reader_reshape_tile_mapped_pipelined.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    TensorAccessorArgs(*map_buf).append_to(reader.compile_time_args);
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.named_compile_time_args = {
        {"map_page_bytes", layout.page_bytes()},
        {"tile_bytes", tile_bytes},
        {"map_cb", kCbMap},
        {"input_cb", kCbMappedInput},
        // Half-depth batches keep a slot free for the tile the writer still holds across rows.
        {"read_batch", std::max<uint32_t>(1, input_depth / 2)},
        {"input_cb_depth", input_depth},
    };
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = op_kernel("writer_reshape_tile_mapped.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.named_compile_time_args = {
        {"tile_bytes", tile_bytes},
        {"max_entries", layout.entries_per_page},
        {"element_bytes", static_cast<uint32_t>(input.element_size())},
        {"map_cb", kCbMap},
        {"input_cb", kCbMappedInput},
        {"output_cb", kCbScratch},
        {"sliced", static_cast<uint32_t>(layout.sliced)},
    };
    writer.config = WriterConfigDescriptor{};

    reader.emplace_common_runtime_args({in_buf, map_buf});
    writer.emplace_common_runtime_args({out_buf});
    const auto cores = corerange_to_cores(all_cores, num_cores, false);
    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t end = start + (group1.contains(core) ? rows1 : rows2);
        reader.emplace_runtime_args(core, {start, end});
        writer.emplace_runtime_args(core, {start, end});
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

    // Selection sends a call here unless the segment map is the live choice, so a call nothing fits reaches the error.
    const auto chosen = rcg::live_tile_rows_choice(input, out_shape, operation_attributes.plan.tile_rows);
    TT_FATAL(
        chosen.has_value(),
        "ReshapeCodegen: neither a tile-row plan nor the segment map fits the {} B of L1 free per core below the "
        "lowest L1 allocation; free L1 tensors or use the native reshape",
        ttnn::operations::data_movement::get_max_l1_space(input));
    auto plan = rcg::build_tile_rows_plan(in_shape, out_shape, *chosen);
    const rcg::TileRowsLayout layout = plan.layout;
    TT_ASSERT(tensor_return_value.buffer()->num_pages() == layout.num_output_pages);

    const auto plan_owner = upload_table(std::move(plan.table), layout.num_batches, layout.page_words, input);
    return table_workload(
        plan_owner,
        build_tile_rows(
            layout, chosen->staged_batches, tile_bytes, input, plan_owner->buffer(), tensor_return_value, grid),
        tensor_coords);
}

tt::tt_metal::WorkloadDescriptor ReshapeCodegenTileMappedProgramFactory::create_workload_descriptor(
    const ReshapeCodegenParams& operation_attributes,
    const ReshapeCodegenInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const Tensor& input = tensor_args.input;
    const auto grid = rcg::reshape_core_grid(input, operation_attributes.sub_core_grids);
    const auto num_cores = static_cast<uint32_t>(grid.num_cores());
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(datatype_to_dataformat_converter(input.dtype()));

    // Serves the segment-map route and the fallback for tile-rows plans that do not fit live L1.
    const auto chosen =
        rcg::live_tile_choice(input, operation_attributes.logical_output_shape, num_cores, operation_attributes.plan);
    const auto* mapped = chosen.has_value() ? std::get_if<rcg::TileMappedChoice>(&*chosen) : nullptr;
    TT_FATAL(
        mapped != nullptr,
        "ReshapeCodegen: no segment map fits the {} B of L1 free per core below the lowest L1 allocation; free L1 "
        "tensors or use the native reshape",
        ttnn::operations::data_movement::get_max_l1_space(input));
    const rcg::TileMappedLayout& layout = mapped->layout;
    TT_ASSERT(tensor_return_value.buffer()->num_pages() == layout.num_output_pages);

    auto plan = rcg::build_tile_mapped_plan(
        input.logical_shape(),
        operation_attributes.logical_output_shape,
        static_cast<uint32_t>(input.element_size()),
        tile_bytes,
        num_cores,
        layout.sliced);
    const auto map_owner = upload_table(std::move(plan.table), layout.num_rows, layout.page_words(), input);
    return table_workload(
        map_owner,
        build_tile_mapped(
            layout, mapped->input_depth, tile_bytes, input, map_owner->buffer(), tensor_return_value, grid),
        tensor_coords);
}

}  // namespace ttnn::prim

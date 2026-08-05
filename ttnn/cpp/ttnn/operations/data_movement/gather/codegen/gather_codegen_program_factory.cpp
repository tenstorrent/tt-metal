// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "gather_codegen_device_operation.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kCbInput = tt::CBIndex::c_0;
constexpr uint32_t kCbIndex = tt::CBIndex::c_1;
constexpr uint32_t kCbOutput = tt::CBIndex::c_2;

constexpr const char* kReaderInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_reader.cpp";
constexpr const char* kWriterInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_writer.cpp";
constexpr const char* kReaderTiled =
    "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_reader_tiled.cpp";
constexpr const char* kWriterTiled =
    "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_writer_tiled.cpp";
constexpr const char* kReaderStreaming =
    "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_reader_streaming.cpp";
constexpr const char* kWriterStreaming =
    "ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_writer_streaming.cpp";

// The device's real per-core CB ceiling, standing in for the Python builders' fixed USABLE_L1.
uint64_t gather_usable_l1(const Tensor& input_tensor) {
    auto* device = input_tensor.device();
    return static_cast<uint64_t>(device->l1_size_per_core()) -
           device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

CoreRangeSet gather_core_grid(IDevice* device, const std::optional<CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        return sub_core_grids.value();
    }
    const auto grid_size = device->compute_with_storage_grid_size();
    return num_cores_to_corerangeset(grid_size.x * grid_size.y, grid_size, /*row_wise=*/true);
}

struct CoreSplit {
    uint32_t num_cores;
    CoreRangeSet core_range;
    CoreRangeSet group1;
    CoreRangeSet group2;
    uint32_t work_per_core_1;
    uint32_t work_per_core_2;
};

// split_cores(device, total_work, core_ranges=sub_core_grids)[0] equivalent: a rectangular,
// row-wise work split over either the caller's sub_core_grids or the full compute grid.
CoreSplit split_gather_work(IDevice* device, const std::optional<CoreRangeSet>& sub_core_grids, uint32_t total_work) {
    const auto core_grid = gather_core_grid(device, sub_core_grids);
    const auto [num_cores, core_range, group1, group2, wpc1, wpc2] =
        tt::tt_metal::split_work_to_cores(core_grid, total_work, /*row_wise=*/true);
    return CoreSplit{num_cores, core_range, group1, group2, wpc1, wpc2};
}

CBDescriptor make_tile_cb(uint32_t cb_id, const Tensor& tensor, uint32_t depth, const CoreRangeSet& core_range) {
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    const uint32_t page_size = tensor.buffer()->aligned_page_size();
    return CBDescriptor{
        .total_size = depth * page_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    };
}

}  // namespace

GatherGeometry compute_gather_geometry(const Tensor& input_tensor, const Tensor& input_index_tensor) {
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();

    const auto padded_input = input_tensor.padded_shape();
    const auto padded_index = input_index_tensor.padded_shape();
    const auto& index_shape = input_index_tensor.logical_shape();

    uint32_t Ht = 1;
    for (uint32_t i = 0; i + 1 < padded_index.rank(); ++i) {
        Ht *= padded_index[i];
    }
    Ht /= tile_height;

    const uint32_t Wt_input = padded_input[-1] / tile_width;
    const uint32_t Wt_index = padded_index[-1] / tile_width;

    const uint32_t index_h_mod = index_shape[-2] % tile_height;
    const uint32_t index_valid_h_last = index_h_mod != 0 ? index_h_mod : tile_height;
    const uint32_t index_w_mod = index_shape[-1] % tile_width;
    const uint32_t index_valid_w_last = index_w_mod != 0 ? index_w_mod : tile_width;
    const uint32_t index_ht_per_batch = padded_index[-2] / tile_height;

    return GatherGeometry{Ht, Wt_input, Wt_index, index_valid_h_last, index_valid_w_last, index_ht_per_batch};
}

bool gather_interleaved_fits_l1(
    const Tensor& input_tensor, const Tensor& input_index_tensor, uint32_t Wt_input, uint32_t Wt_index) {
    // Output dtype always equals input dtype (compute_output_specs) and this predicate is only ever
    // consulted for the non-sharded DRAM-interleaved scope supported_by_codegen() admits, so the
    // output tile's aligned page size equals the input tensor's.
    const uint64_t input_page = input_tensor.buffer()->aligned_page_size();
    const uint64_t index_page = input_index_tensor.buffer()->aligned_page_size();
    const uint64_t output_page = input_page;
    const uint64_t footprint =
        static_cast<uint64_t>(Wt_input) * input_page + index_page + std::max<uint64_t>(4, Wt_index) * output_page;
    // ops/gather/gather.py::_interleaved_fits_l1 additionally short-circuits
    // Wt_input <= _WT_ROWBUF_FLOOR (60) to true. That short-circuit is deliberately not carried
    // over: the gathered axis is the one axis whose index extent is NOT bounded by the input
    // extent, so a narrow row with a very wide index clears the floor while its
    // max(4, Wt_index)-deep output CB does not fit, and the row-buffered plan would fail CB
    // allocation instead of reaching the streaming one. Below the floor with a proportionate index
    // the footprint always fits, so selection is unchanged wherever the floor was reachable.
    return footprint <= gather_usable_l1(input_tensor);
}

bool gather_min_plan_fits_l1(const Tensor& input_tensor, const Tensor& input_index_tensor) {
    // Smallest plan any of the three factories can be built with: the streaming factory's floor of
    // two input pages (gather_streaming_chunk_tiles' minimum), one index page and one output page.
    // Below this there is no depth left to scale down to, so the routing gate must send the call to
    // native instead of failing during program creation.
    const uint64_t input_page = input_tensor.buffer()->aligned_page_size();
    const uint64_t index_page = input_index_tensor.buffer()->aligned_page_size();
    const uint64_t output_page = input_page;
    return 2 * input_page + index_page + output_page <= gather_usable_l1(input_tensor);
}

uint32_t gather_streaming_chunk_tiles(const Tensor& input_tensor, const Tensor& input_index_tensor, uint32_t Wt_input) {
    const uint64_t usable_l1 = gather_usable_l1(input_tensor);
    const uint64_t input_page = input_tensor.buffer()->aligned_page_size();
    const uint64_t index_page = input_index_tensor.buffer()->aligned_page_size();
    // Output dtype always equals input dtype, same as gather_interleaved_fits_l1's reasoning.
    const uint64_t fixed_pages = index_page + input_page;
    const uint64_t affordable = usable_l1 > fixed_pages ? (usable_l1 - fixed_pages) / input_page : 0;
    // Two keeps the reader's DRAM reads overlapping its scan when L1 is too tight for more; the row
    // length is the ceiling because a deeper block would just hold pages no index can select.
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(affordable, 2), Wt_input));
}

tt::tt_metal::ProgramDescriptor GatherCodegenProgramFactoryInterleaved::create_descriptor(
    const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.input_index_tensor;
    const GatherGeometry geometry{
        attributes.Ht,
        attributes.Wt_input,
        attributes.Wt_index,
        attributes.index_valid_h_last,
        attributes.index_valid_w_last,
        attributes.index_ht_per_batch};

    auto* device = in_t.device();
    const auto split = split_gather_work(device, attributes.sub_core_grids, geometry.Ht);

    const uint32_t tile_width = in_t.tensor_spec().tile().get_width();
    const uint32_t tile_height = in_t.tensor_spec().tile().get_height();

    ProgramDescriptor desc;
    desc.cbs.push_back(make_tile_cb(kCbInput, in_t, geometry.Wt_input, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbIndex, index_t, 1, split.core_range));
    desc.cbs.push_back(
        make_tile_cb(kCbOutput, output_tensor, std::max<uint32_t>(4, geometry.Wt_index), split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbIndex,
        kCbOutput,
        geometry.Wt_input,
        geometry.Wt_index,
        split.num_cores,
        geometry.index_valid_h_last,
        geometry.index_valid_w_last,
        geometry.index_ht_per_batch,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderInterleaved;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {
        kCbInput, kCbOutput, geometry.Wt_input, geometry.Wt_index, split.num_cores};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterInterleaved;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core RT follows the ORDINAL convention (spec.py's `_ordinal_rt`): a sequential counter over
    // assigned cores in iter_cores order (group1 rows, then group2 rows), NOT the per-core work
    // offset. Kernel ABI: reader [index_addr, n, tile_w, tile_h, core_id], writer
    // [in_addr, out_addr, n, core_id].
    uint32_t id = 0;
    for (const auto& [group, n] :
         {std::make_pair(split.group1, split.work_per_core_1), std::make_pair(split.group2, split.work_per_core_2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                if (n > 0) {
                    reader_desc.emplace_runtime_args(core, {index_t.buffer(), n, tile_width, tile_height, id});
                    writer_desc.emplace_runtime_args(core, {in_t.buffer(), output_tensor.buffer(), n, id});
                } else {
                    reader_desc.emplace_runtime_args(core, {0u, 0u, tile_width, tile_height, id});
                    writer_desc.emplace_runtime_args(core, {0u, 0u, 0u, id});
                }
                id++;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

tt::tt_metal::ProgramDescriptor GatherCodegenProgramFactoryTiled::create_descriptor(
    const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.input_index_tensor;
    const GatherGeometry geometry{
        attributes.Ht,
        attributes.Wt_input,
        attributes.Wt_index,
        attributes.index_valid_h_last,
        attributes.index_valid_w_last,
        attributes.index_ht_per_batch};
    const uint32_t total_work = geometry.Ht * geometry.Wt_index;

    auto* device = in_t.device();
    const auto split = split_gather_work(device, attributes.sub_core_grids, total_work);

    const uint32_t tile_width = in_t.tensor_spec().tile().get_width();
    const uint32_t tile_height = in_t.tensor_spec().tile().get_height();

    ProgramDescriptor desc;
    // Identical three-CB footprint to the Interleaved factory (see build_gather_tiled_factory's
    // docstring): `gather_interleaved_fits_l1` remains the correct admission test for this factory.
    desc.cbs.push_back(make_tile_cb(kCbInput, in_t, geometry.Wt_input, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbIndex, index_t, 1, split.core_range));
    desc.cbs.push_back(
        make_tile_cb(kCbOutput, output_tensor, std::max<uint32_t>(4, geometry.Wt_index), split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbIndex,
        kCbOutput,
        geometry.Wt_input,
        geometry.Wt_index,
        split.num_cores,
        geometry.index_valid_h_last,
        geometry.index_valid_w_last,
        geometry.index_ht_per_batch,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderTiled;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {
        kCbInput, kCbOutput, geometry.Wt_input, geometry.Wt_index, split.num_cores};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterTiled;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core RT is the CONTIGUOUS [start, n) output-tile range emit_per_core_rt assigns (the
    // default `[n, start]` work-offset convention, unlike Interleaved/Streaming's core ordinal).
    // Kernel ABI: reader [index_addr, start, n, tile_w, tile_h], writer [in_addr, out_addr, start, n].
    uint32_t start = 0;
    for (const auto& [group, n] :
         {std::make_pair(split.group1, split.work_per_core_1), std::make_pair(split.group2, split.work_per_core_2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                if (n > 0) {
                    reader_desc.emplace_runtime_args(core, {index_t.buffer(), start, n, tile_width, tile_height});
                    writer_desc.emplace_runtime_args(core, {in_t.buffer(), output_tensor.buffer(), start, n});
                } else {
                    reader_desc.emplace_runtime_args(core, {0u, start, 0u, tile_width, tile_height});
                    writer_desc.emplace_runtime_args(core, {0u, 0u, start, 0u});
                }
                start += n;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

tt::tt_metal::ProgramDescriptor GatherCodegenProgramFactoryStreaming::create_descriptor(
    const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.input_index_tensor;
    const GatherGeometry geometry{
        attributes.Ht,
        attributes.Wt_input,
        attributes.Wt_index,
        attributes.index_valid_h_last,
        attributes.index_valid_w_last,
        attributes.index_ht_per_batch};

    auto* device = in_t.device();
    const auto split = split_gather_work(device, attributes.sub_core_grids, geometry.Wt_index);

    const uint32_t tile_width = in_t.tensor_spec().tile().get_width();
    const uint32_t tile_height = in_t.tensor_spec().tile().get_height();
    const uint32_t chunk_tiles = gather_streaming_chunk_tiles(in_t, index_t, geometry.Wt_input);

    ProgramDescriptor desc;
    // The input CB holds one chunk_tiles-deep block of the row rather than the whole row, which is
    // what makes this the fallback when gather_interleaved_fits_l1() rejects the row-buffered plan.
    // Both kernels walk ceil(Wt_input / chunk_tiles) blocks of exactly chunk_tiles pages (the tail
    // padded), so the CB is emptied every block and never wraps mid-block.
    desc.cbs.push_back(make_tile_cb(kCbInput, in_t, chunk_tiles, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbIndex, index_t, 1, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbOutput, output_tensor, 1, split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbIndex,
        kCbOutput,
        geometry.Ht,
        geometry.Wt_input,
        geometry.Wt_index,
        split.num_cores,
        geometry.index_valid_h_last,
        geometry.index_valid_w_last,
        geometry.index_ht_per_batch,
        chunk_tiles,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderStreaming;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {
        kCbInput, kCbOutput, geometry.Ht, geometry.Wt_input, geometry.Wt_index, split.num_cores, chunk_tiles};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterStreaming;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Ordinal RT convention, same as Interleaved (work is split by Wt_index; each core streams its
    // strided columns across all Ht rows -- see gather_reader_streaming.cpp).
    uint32_t id = 0;
    for (const auto& [group, n] :
         {std::make_pair(split.group1, split.work_per_core_1), std::make_pair(split.group2, split.work_per_core_2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                if (n > 0) {
                    reader_desc.emplace_runtime_args(core, {index_t.buffer(), n, tile_width, tile_height, id});
                    writer_desc.emplace_runtime_args(core, {in_t.buffer(), output_tensor.buffer(), n, id});
                } else {
                    reader_desc.emplace_runtime_args(core, {0u, 0u, tile_width, tile_height, id});
                    writer_desc.emplace_runtime_args(core, {0u, 0u, 0u, id});
                }
                id++;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim

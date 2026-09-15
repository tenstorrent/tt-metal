// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
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

// The device's STATIC per-core CB budget: the whole allocator-managed L1 window, with no regard for
// what is live in it.
//
// This is the only budget a routing gate may plan against. supported_by_codegen() is evaluated twice
// in one dispatch -- ttnn::gather()'s router and
// GatherCodegenDeviceOperation::validate_on_program_cache_miss -- with the op's own
// create_output_tensors() in between, and those two sites are consistent with each other only
// because the answer cannot move: budget the gate against live occupancy and an L1 output lowers the
// ceiling between them, so routing admits the call and validate then TT_FATALs on the tensor it is
// already committed to.
uint64_t gather_static_l1(const Tensor& input_tensor) {
    auto* device = input_tensor.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t ceiling = static_cast<uint64_t>(device->l1_size_per_core());
    return ceiling > base ? ceiling - base : 0;
}

// The device's real per-core CB ceiling. Confined to the program factory, which reads it once per
// cache miss with nothing moving underneath it.
//
// L1 tensors are allocated downward from the top of L1 while static CBs stack upward from the
// allocator base, so any live L1 buffer -- this call's own output when output_mem_config asks for
// L1 -- is the true ceiling: Program::validate_circular_buffer_region rejects a CB region that
// ends above the lowest occupied L1 address, so a plan built against the static budget alone would
// fail program creation instead of scaling down to one that fits.
//
// The frontier is read here at program-creation time and baked into the resulting program, so it
// has to reach the program-cache key as well: validate_circular_buffer_region re-reads the frontier
// on every enqueue, cache hit included, and checks it against the region the cached program already
// carries, so a plan cached under a clear frontier throws once anything lowers it.
// GatherCodegenDeviceOperation::compute_program_hash therefore folds in the plan this budget
// derives -- the selected factory and, for streaming, the block depth -- rather than the frontier
// itself, which would miss on every unrelated allocation.
uint64_t gather_usable_l1(const Tensor& input_tensor) {
    auto* device = input_tensor.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint64_t usable = gather_static_l1(input_tensor);
    if (const auto lowest_l1_buffer = device->lowest_occupied_compute_l1_address(); lowest_l1_buffer.has_value()) {
        const uint64_t frontier = static_cast<uint64_t>(lowest_l1_buffer.value());
        usable = std::min(usable, frontier > base ? frontier - base : 0);
    }
    return usable;
}

// Packs `nc` cores into at most two CoreRanges by treating x as the row index and grid.y as the row
// length, so dispatch carries 1-2 range entries rather than one per core.
CoreRangeSet gather_rect_core_range_set(uint32_t nc, const CoreCoord& grid) {
    const uint32_t cols = static_cast<uint32_t>(grid.y);
    const uint32_t full_rows = nc / cols;
    const uint32_t remaining = nc % cols;
    std::vector<CoreRange> ranges;
    if (full_rows > 0) {
        ranges.emplace_back(CoreCoord(0, 0), CoreCoord(full_rows - 1, cols - 1));
    }
    if (remaining > 0) {
        ranges.emplace_back(CoreCoord(full_rows, 0), CoreCoord(full_rows, remaining - 1));
    }
    return CoreRangeSet(std::move(ranges));
}

struct CoreSplit {
    uint32_t num_cores;
    CoreRangeSet core_range;
    CoreRangeSet group1;
    CoreRangeSet group2;
    uint32_t work_per_core_1;
    uint32_t work_per_core_2;
};

// An explicit sub_core_grids is authoritative and goes straight to the splitter; otherwise the
// candidate set is exactly min(total_work, device cores) cores, so the splitter always sees
// units >= candidate cores and hands back the candidate set verbatim.
//
// row_wise=false is load-bearing rather than cosmetic: the row-buffered/streaming kernels stride by
// num_cores from their ordinal, so an ordinal at or above the extra-work core count must never
// receive a second work unit -- which holds only while gather_assigned_cores() numbers ordinals in
// the same order the splitter carved the extra-work group, and that order is column-major.
CoreSplit split_gather_work(IDevice* device, const std::optional<CoreRangeSet>& sub_core_grids, uint32_t total_work) {
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t device_cores = static_cast<uint32_t>(grid.x * grid.y);
    const CoreRangeSet candidate = sub_core_grids.has_value()
                                       ? sub_core_grids.value()
                                       : gather_rect_core_range_set(std::min(total_work, device_cores), grid);
    const auto [num_cores, core_range, group1, group2, wpc1, wpc2] =
        tt::tt_metal::split_work_to_cores(candidate, total_work, /*row_wise=*/false);
    return CoreSplit{num_cores, core_range, group1, group2, wpc1, wpc2};
}

// Yield each assigned core with its work count, in the order the splitter itself consumed the core
// set: its ranges in the order they were supplied, column-major within each. Both the per-core
// ordinal (row-buffered, streaming) and the contiguous [start, n) offsets (tiled) are numbered in
// exactly this order.
//
// split_work_to_cores() carves the extra-work group off the FRONT of that order, so an enumeration
// that disagrees with it -- which a multi-range sub_core_grids makes possible -- hands the extra
// unit to a core whose ordinal is past the extra-work count: the strided kernels then drop one
// tile-row and address another past the end of the tensor.
std::vector<std::pair<CoreCoord, uint32_t>> gather_assigned_cores(const CoreSplit& split, uint32_t total_work) {
    std::vector<std::pair<CoreCoord, uint32_t>> assigned;
    assigned.reserve(split.num_cores);
    uint32_t emitted = 0;
    for (const auto& core : corerange_to_cores(split.core_range, std::nullopt, /*row_wise=*/false)) {
        uint32_t work = 0;
        if (split.group1.contains(core)) {
            work = split.work_per_core_1;
        } else if (split.group2.contains(core)) {
            work = split.work_per_core_2;
        } else {
            continue;
        }
        work = std::min(work, total_work - emitted);
        emitted += work;
        assigned.emplace_back(core, work);
    }
    return assigned;
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

    const auto& padded_input = input_tensor.padded_shape();
    const auto& padded_index = input_index_tensor.padded_shape();
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
        static_cast<uint64_t>(Wt_input) * input_page + index_page + gather_output_cb_tiles(Wt_index) * output_page;
    // Every term is checked, with no short-circuit for a narrow row: the gathered axis is the one
    // axis whose index extent is NOT bounded by the input extent, so a narrow row with a very wide
    // index would clear any Wt_input floor while its gather_output_cb_tiles(Wt_index)-deep output CB does not fit,
    // and the row-buffered plan would fail CB allocation instead of reaching the streaming one.
    return footprint <= gather_usable_l1(input_tensor);
}

bool gather_min_plan_fits_l1(const Tensor& input_tensor, const Tensor& input_index_tensor) {
    // Smallest plan any of the three factories can be built with: the streaming factory's floor of
    // two input pages (gather_streaming_chunk_tiles' minimum), one index page and one output page.
    // Below this there is no depth left to scale down to, so the routing gate must send the call to
    // native instead of failing during program creation.
    //
    // Budgeted against the STATIC window rather than the live one, because this is a routing gate
    // (see gather_static_l1). The residual case that leaves -- a device whose live L1 cannot even
    // seat these four pages -- has nothing to route to either: native's own CBs are fixed at
    // Wt_input input pages, so it fails the same allocation harder.
    const uint64_t input_page = input_tensor.buffer()->aligned_page_size();
    const uint64_t index_page = input_index_tensor.buffer()->aligned_page_size();
    const uint64_t output_page = input_page;
    return 2 * input_page + index_page + output_page <= gather_static_l1(input_tensor);
}

uint32_t gather_streaming_chunk_tiles(const Tensor& input_tensor, const Tensor& input_index_tensor, uint32_t Wt_input) {
    const uint64_t usable_l1 = gather_usable_l1(input_tensor);
    const uint64_t input_page = input_tensor.buffer()->aligned_page_size();
    const uint64_t index_page = input_index_tensor.buffer()->aligned_page_size();
    // Output dtype always equals input dtype, same as gather_interleaved_fits_l1's reasoning.
    const uint64_t fixed_pages = index_page + input_page;
    const uint64_t affordable = usable_l1 > fixed_pages ? (usable_l1 - fixed_pages) / input_page : 0;
    // Deepest block L1 affords, capped at the row. Two is the floor (it keeps the reader's DRAM
    // reads overlapping its scan when L1 is too tight for more) and is what
    // gather_min_plan_fits_l1() gates against.
    const uint32_t max_resident =
        static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(affordable, 2), Wt_input));
    // What the resident depth actually buys is the BLOCK COUNT: both streaming kernels rescan the
    // whole index tile once per block (gather_reader_streaming.cpp's chunk loop), so the scalar cost
    // per output tile is n_chunks * TILE_HW and nothing else about the depth changes it. The writer,
    // however, pushes a full chunk_tiles-deep block every time and pads a short tail by re-reading
    // the row's last page (gather_writer_streaming.cpp's `page_w` clamp), so the DRAM cost is
    // n_chunks * chunk_tiles pages per row -- not Wt_input. Spreading the row evenly over the same
    // block count keeps the scan count identical and drops the padding re-reads: a Wt_input=1000 row
    // that needs two blocks then costs 1000 input pages per row instead of 2 * ceiling.
    //
    // Deriving the depth from the block count also makes the page count independent of the budget,
    // so a larger budget can only lower the scan count, never raise it.
    const uint32_t n_chunks = (Wt_input + max_resident - 1) / max_resident;
    return (Wt_input + n_chunks - 1) / n_chunks;
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
        make_tile_cb(kCbOutput, output_tensor, gather_output_cb_tiles(geometry.Wt_index), split.core_range));

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
        kCbInput,
        kCbOutput,
        geometry.Wt_input,
        geometry.Wt_index,
        split.num_cores,
        gather_output_cb_tiles(geometry.Wt_index),
        kGatherWriteBatchTiles};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterInterleaved;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core RT follows the ORDINAL convention: a sequential counter over assigned cores in
    // gather_assigned_cores() order, NOT the per-core work offset. Kernel ABI:
    // reader [index_addr, n, tile_w, tile_h, core_id], writer [in_addr, out_addr, n, core_id].
    uint32_t id = 0;
    for (const auto& [core, n] : gather_assigned_cores(split, geometry.Ht)) {
        reader_desc.emplace_runtime_args(core, {index_t.buffer(), n, tile_width, tile_height, id});
        writer_desc.emplace_runtime_args(core, {in_t.buffer(), output_tensor.buffer(), n, id});
        id++;
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
    // Identical three-CB footprint to the Interleaved factory, so `gather_interleaved_fits_l1` is
    // the correct admission test for this factory too.
    desc.cbs.push_back(make_tile_cb(kCbInput, in_t, geometry.Wt_input, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbIndex, index_t, 1, split.core_range));
    desc.cbs.push_back(
        make_tile_cb(kCbOutput, output_tensor, gather_output_cb_tiles(geometry.Wt_index), split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbIndex,
        kCbOutput,
        geometry.Wt_input,
        geometry.Wt_index,
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
        kCbInput,
        kCbOutput,
        geometry.Wt_input,
        geometry.Wt_index,
        gather_output_cb_tiles(geometry.Wt_index),
        kGatherWriteBatchTiles};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterTiled;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core RT is the CONTIGUOUS [start, n) output-tile range, unlike Interleaved/Streaming's
    // core ordinal.
    // Kernel ABI: reader [index_addr, start, n, tile_w, tile_h], writer [in_addr, out_addr, start, n].
    uint32_t start = 0;
    for (const auto& [core, n] : gather_assigned_cores(split, total_work)) {
        reader_desc.emplace_runtime_args(core, {index_t.buffer(), start, n, tile_width, tile_height});
        writer_desc.emplace_runtime_args(core, {in_t.buffer(), output_tensor.buffer(), start, n});
        start += n;
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
    // Both kernels walk ceil(Wt_input / chunk_tiles) blocks of exactly chunk_tiles pages, so the CB
    // is emptied every block and never wraps mid-block.
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
    for (const auto& [core, n] : gather_assigned_cores(split, geometry.Wt_index)) {
        reader_desc.emplace_runtime_args(core, {index_t.buffer(), n, tile_width, tile_height, id});
        writer_desc.emplace_runtime_args(core, {in_t.buffer(), output_tensor.buffer(), n, id});
        id++;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "scatter_codegen_device_operation.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

namespace {

// Deliberately not gather's map (input on c_0, output on c_2): the kernels are written against this
// one, and a watcher dump of a scatter program reads with these ids.
constexpr uint32_t kCbOutput = tt::CBIndex::c_0;
constexpr uint32_t kCbIndex = tt::CBIndex::c_1;
constexpr uint32_t kCbSrc = tt::CBIndex::c_2;
constexpr uint32_t kCbInput = tt::CBIndex::c_3;
constexpr uint32_t kCbFp32Temp = tt::CBIndex::c_4;

constexpr const char* kReaderInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_reader.cpp";
constexpr const char* kWriterInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_writer.cpp";
constexpr const char* kReaderStreaming =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_reader_streaming.cpp";
constexpr const char* kWriterStreaming =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_writer_streaming.cpp";
constexpr const char* kReaderRm =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_reader_rm.cpp";
constexpr const char* kWriterRm =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_writer_rm.cpp";
constexpr const char* kReaderBf16ReduceRm =
    "ttnn/cpp/ttnn/operations/data_movement/scatter/codegen/kernels/scatter_reader_bf16_reduce_rm.cpp";

// Default RM index/src streaming chunk before scaling to the live L1 frontier: the generator's
// default, 48 KiB per core for an int32 index with bf16 sources, small next to the resident sticks.
constexpr uint32_t kRmDefaultChunkElems = 8192;
// A chunk boundary's byte offset is its element count times the element size, so a multiple of 32
// elements of the 2- and 4-byte dtypes in scope is at least 64 B: NOC-aligned on Wormhole (32 B)
// and Blackhole (64 B) alike.
constexpr uint32_t kRmChunkAlign = 32;
// Smallest chunk the scaling hands out, and the floor scatter_rm_min_plan_fits_l1() checks against.
constexpr uint32_t kRmMinChunkElems = 32;

std::vector<uint32_t> scatter_page_dims(const ttnn::Shape& shape, bool tiled) {
    const int64_t rank = static_cast<int64_t>(shape.rank());
    if (rank == 0) {
        return {1};
    }
    if (tiled) {
        if (rank == 1) {
            return {1};
        }
        std::vector<uint32_t> dims;
        dims.reserve(static_cast<size_t>(rank) - 1);
        for (int64_t i = 0; i < rank - 2; ++i) {
            dims.push_back(shape[i]);
        }
        dims.push_back((shape[rank - 2] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT);
        return dims;
    }
    if (rank == 1) {
        return {1};
    }
    std::vector<uint32_t> dims;
    dims.reserve(static_cast<size_t>(rank) - 1);
    for (int64_t i = 0; i < rank - 1; ++i) {
        dims.push_back(shape[i]);
    }
    return dims;
}

struct CoreSplit {
    uint32_t num_cores;
    CoreRangeSet core_range;
    CoreRangeSet group1;
    CoreRangeSet group2;
    uint32_t work_per_core_1;
    uint32_t work_per_core_2;
};

// An explicit sub_core_grids is authoritative; otherwise the candidate set is min(total_work, device
// cores). row_wise=false is load-bearing: the ordinal-numbered kernels stride by num_cores from their
// assigned ordinal, so ordinal numbering must agree with the order split_work_to_cores() carved its
// extra-work group from (its ranges in stored order, column-major within each). Sweeping the device
// grid directly instead of walking this same assigned set would silently disagree whenever the
// grid's natural sweep order differs from how the splitter grouped its ranges.
CoreSplit split_scatter_work(
    const MeshDevice& device, const std::optional<CoreRangeSet>& sub_core_grids, uint32_t total_work) {
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t device_cores = static_cast<uint32_t>(grid.x * grid.y);
    CoreRangeSet candidate;
    if (sub_core_grids.has_value()) {
        candidate = sub_core_grids.value();
    } else {
        const uint32_t cols = static_cast<uint32_t>(grid.y);
        const uint32_t nc = std::min(total_work, device_cores);
        const uint32_t full_rows = nc / cols;
        const uint32_t remaining = nc % cols;
        std::vector<CoreRange> ranges;
        if (full_rows > 0) {
            ranges.emplace_back(CoreCoord(0, 0), CoreCoord(full_rows - 1, cols - 1));
        }
        if (remaining > 0) {
            ranges.emplace_back(CoreCoord(full_rows, 0), CoreCoord(full_rows, remaining - 1));
        }
        candidate = CoreRangeSet(std::move(ranges));
    }
    const auto [num_cores, core_range, group1, group2, wpc1, wpc2] =
        tt::tt_metal::split_work_to_cores(candidate, total_work, /*row_wise=*/false);
    return CoreSplit{num_cores, core_range, group1, group2, wpc1, wpc2};
}

// Yields each assigned core with its work count, in the same order split_work_to_cores() consumed the
// core set (its ranges in stored order, column-major within each). Callers number this either as a
// sequential ordinal (interleaved/streaming) or as a running [start, n) offset (ROW_MAJOR).
std::vector<std::pair<CoreCoord, uint32_t>> scatter_assigned_cores(const CoreSplit& split, uint32_t total_work) {
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

// CB sizes reach the device in 16 B words, so a page that is not a multiple of 16 B would leave the
// kernel's view of the buffer short of what the host wrote into it.
constexpr uint32_t kCbWordBytes = 16;

CBDescriptor make_rm_cb(
    uint32_t cb_id,
    tt::tt_metal::DataType dtype,
    uint32_t raw_page_size,
    uint32_t depth,
    const CoreRangeSet& core_range) {
    const uint32_t page_size = tt::align(raw_page_size, kCbWordBytes);
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
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

std::vector<uint32_t> to_vector(const ScatterPageMap& page_map) { return {page_map.begin(), page_map.end()}; }

}  // namespace

uint32_t scatter_page_rank(const ttnn::Shape& shape, bool tiled) {
    return static_cast<uint32_t>(scatter_page_dims(shape, tiled).size());
}

ScatterPageMap compute_scatter_page_map(const ttnn::Shape& input_shape, const ttnn::Shape& index_shape, bool tiled) {
    const auto input_dims = scatter_page_dims(input_shape, tiled);
    const auto index_dims = scatter_page_dims(index_shape, tiled);
    // Runs ahead of validate_on_program_cache_miss (build_scatter_codegen_params), which owns the
    // equal-rank and max-rank checks; clamp so an operand validate will reject cannot index past
    // the fixed-width map here.
    const uint32_t rank = static_cast<uint32_t>(
        std::min({input_dims.size(), index_dims.size(), static_cast<size_t>(kScatterMaxPageRank)}));
    ScatterPageMap page_map{};
    page_map[0] = rank;
    for (uint32_t i = 0; i < kScatterMaxPageRank; ++i) {
        page_map[1 + i] = i < rank ? input_dims[i] : 1;
        page_map[1 + kScatterMaxPageRank + i] = i < rank ? index_dims[i] : 1;
    }
    return page_map;
}

ScatterTileGeometry compute_scatter_tile_geometry(
    const Tensor& input_tensor, const Tensor& index_tensor, const Tensor& src_tensor) {
    const uint32_t tile_h = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_w = input_tensor.tensor_spec().tile().get_width();

    const auto& padded_input = input_tensor.padded_shape();
    const auto& padded_index = index_tensor.padded_shape();
    const auto& padded_src = src_tensor.padded_shape();
    const auto& logical_index = index_tensor.logical_shape();

    uint32_t Ht = 1;
    for (uint32_t i = 0; i + 1 < padded_input.rank(); ++i) {
        Ht *= padded_input[i];
    }
    Ht /= tile_h;

    const uint32_t Wt_output = padded_input[-1] / tile_w;
    const uint32_t Wt_index = padded_index[-1] / tile_w;

    const uint32_t idx_h_mod = logical_index[-2] % tile_h;
    const uint32_t idx_valid_h_last = idx_h_mod != 0 ? idx_h_mod : tile_h;
    const uint32_t idx_w_mod = logical_index[-1] % tile_w;
    const uint32_t idx_valid_w_last = idx_w_mod != 0 ? idx_w_mod : tile_w;

    const uint32_t Ht_per_batch_index = padded_src[-2] / tile_h;

    return ScatterTileGeometry{
        Ht,
        Wt_output,
        Wt_index,
        input_tensor.logical_shape()[-1],
        idx_valid_h_last,
        idx_valid_w_last,
        Ht_per_batch_index};
}

ScatterRmGeometry compute_scatter_rm_geometry(const Tensor& input_tensor, const Tensor& index_tensor) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto& index_shape = index_tensor.logical_shape();
    uint32_t num_sticks = 1;
    for (uint32_t i = 0; i + 1 < input_shape.rank(); ++i) {
        num_sticks *= input_shape[i];
    }
    return ScatterRmGeometry{num_sticks, input_shape[-1], index_shape[-1]};
}

uint64_t scatter_output_aligned_page_size(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    if (output_tensor.has_value()) {
        return output_tensor->buffer()->aligned_page_size();
    }
    auto* device = input_tensor.device();
    const uint64_t alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    return tt::align(input_tensor.buffer()->page_size(), alignment);
}

uint64_t scatter_rm_stick_page_bytes(const Tensor& tensor, uint32_t stick_elems) {
    auto* device = tensor.device();
    const uint64_t raw_bytes = static_cast<uint64_t>(stick_elems) * tensor.element_size();
    const uint64_t alignment = device->allocator()->get_alignment(tensor.memory_config().buffer_type());
    return tt::align(raw_bytes, alignment);
}

uint64_t scatter_static_l1(const Tensor& input_tensor) {
    auto* device = input_tensor.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t ceiling = static_cast<uint64_t>(device->l1_size_per_core());
    // The bank excludes the L1_SMALL region a device may be opened with, which sits at the top of the
    // span and which the occupancy query below never reports; CBs planned past the bank would overlap
    // it silently.
    const uint64_t bank = device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
    return std::min(ceiling > base ? ceiling - base : 0, bank);
}

uint64_t scatter_usable_l1(const Tensor& input_tensor) {
    auto* device = input_tensor.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint64_t usable = scatter_static_l1(input_tensor);
    if (const auto lowest_l1_buffer = device->lowest_occupied_compute_l1_address(); lowest_l1_buffer.has_value()) {
        const uint64_t frontier = static_cast<uint64_t>(lowest_l1_buffer.value());
        usable = std::min(usable, frontier > base ? frontier - base : 0);
    }
    return usable;
}

bool scatter_interleaved_fits_l1(
    uint64_t l1_budget,
    uint32_t Wt_output,
    uint64_t output_page_bytes,
    uint64_t input_page_bytes,
    uint64_t index_page_bytes,
    uint64_t src_page_bytes) {
    const uint64_t footprint =
        static_cast<uint64_t>(Wt_output) * (output_page_bytes + input_page_bytes) + index_page_bytes + src_page_bytes;
    return footprint <= l1_budget;
}

uint64_t scatter_rm_fixed_bytes(uint64_t input_page_bytes, uint64_t output_page_bytes, bool bf16_reduce) {
    return input_page_bytes + output_page_bytes + (bf16_reduce ? 2 * input_page_bytes : 0);
}

bool scatter_rm_min_plan_fits_l1(
    uint64_t l1_budget, uint64_t input_page_bytes, uint32_t index_elem_size, uint32_t src_elem_size, bool bf16_reduce) {
    // The gate runs before the output exists; supported_execution_controls() pins the output's buffer
    // type to the input's, so the output page is the input's own.
    const uint64_t fixed_bytes = scatter_rm_fixed_bytes(input_page_bytes, input_page_bytes, bf16_reduce);
    const uint64_t min_chunk_bytes = static_cast<uint64_t>(kRmMinChunkElems) * (index_elem_size + src_elem_size);
    return fixed_bytes + min_chunk_bytes <= l1_budget;
}

uint32_t scatter_rm_chunk_elems(
    uint64_t usable_l1,
    uint64_t fixed_bytes,
    uint32_t index_stick_elems,
    uint32_t index_elem_size,
    uint32_t src_elem_size) {
    const uint64_t per_elem_bytes = static_cast<uint64_t>(index_elem_size) + static_cast<uint64_t>(src_elem_size);
    uint32_t chunk = std::min(index_stick_elems, kRmDefaultChunkElems);
    if (fixed_bytes + static_cast<uint64_t>(chunk) * per_elem_bytes > usable_l1) {
        const uint64_t budget = usable_l1 > fixed_bytes ? usable_l1 - fixed_bytes : 0;
        const uint64_t affordable = budget / per_elem_bytes;
        chunk = static_cast<uint32_t>(std::min<uint64_t>(affordable, chunk));
        // Chunking is now actually in play (chunk < index_stick_elems): preserve the 32-byte NOC
        // alignment every chunk-boundary offset needs.
        chunk = (chunk / kRmChunkAlign) * kRmChunkAlign;
        // The forced entry and direct prim callers are gated on the static bank only, and even a routed
        // call's own L1 output seats between routing and this factory, so the frontier read here can
        // be tighter than what was planned against. Rounding an under-budget affordance down to zero
        // would hand the reader kernels a chunk_elems that never advances their `base += chunk_elems`
        // loop, hanging the device instead of failing. Clamp to the smallest NOC-alignment-safe chunk
        // and let Program::validate_circular_buffer_region raise a real allocation error if even that
        // does not fit.
        chunk = std::max(chunk, kRmMinChunkElems);
    }
    return chunk;
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryInterleaved::create_descriptor(
    const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.index_tensor;
    const auto& src_t = tensor_args.src_tensor;

    auto* device = in_t.device();
    const auto split = split_scatter_work(*device, attributes.sub_core_grids, attributes.Ht);

    ProgramDescriptor desc;
    desc.cbs.push_back(make_tile_cb(kCbOutput, output_tensor, attributes.Wt_output, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbIndex, index_t, 1, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbSrc, src_t, 1, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbInput, in_t, attributes.Wt_output, split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbOutput,
        kCbIndex,
        kCbSrc,
        attributes.Wt_output,
        attributes.Wt_index,
        split.num_cores,
        attributes.idx_valid_h_last,
        attributes.idx_valid_w_last,
        attributes.Ht_per_batch_index,
        attributes.output_logical_w,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*src_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderInterleaved;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.named_compile_time_args = {
        {"reduction_mode", attributes.reduction_mode},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {kCbInput, kCbOutput, attributes.Wt_output, split.num_cores};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterInterleaved;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Buffer addresses are common (per-kernel) runtime args, so a program-cache hit rebinds four slots
    // in total rather than four per core. Per-core RT is the sequential assigned-core ORDINAL (0, 1,
    // 2, ...), not the work offset. Kernel ABI: reader common [index_addr, src_addr], per-core [n,
    // tile_w, tile_h, core_id, page_map...]; writer common [in_addr, out_addr], per-core [n, core_id].
    reader_desc.emplace_common_runtime_args({index_t.buffer(), src_t.buffer()});
    writer_desc.emplace_common_runtime_args({in_t.buffer(), output_tensor.buffer()});
    const uint32_t tile_w = in_t.tensor_spec().tile().get_width();
    const uint32_t tile_h = in_t.tensor_spec().tile().get_height();
    const auto page_map_vec = to_vector(attributes.page_map);
    uint32_t id = 0;
    for (const auto& [core, n] : scatter_assigned_cores(split, attributes.Ht)) {
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(n);
        reader_args.push_back(tile_w);
        reader_args.push_back(tile_h);
        reader_args.push_back(id);
        reader_args.append(page_map_vec);
        reader_desc.emplace_runtime_args(core, reader_args);
        writer_desc.emplace_runtime_args(core, {n, id});
        id++;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryStreaming::create_descriptor(
    const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.index_tensor;
    const auto& src_t = tensor_args.src_tensor;

    auto* device = in_t.device();
    const auto split = split_scatter_work(*device, attributes.sub_core_grids, attributes.Wt_output);

    ProgramDescriptor desc;
    desc.cbs.push_back(make_tile_cb(kCbOutput, output_tensor, 2, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbIndex, index_t, 1, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbSrc, src_t, 1, split.core_range));
    desc.cbs.push_back(make_tile_cb(kCbInput, in_t, 2, split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbOutput,
        kCbIndex,
        kCbSrc,
        attributes.Ht,
        attributes.Wt_output,
        attributes.Wt_index,
        split.num_cores,
        attributes.idx_valid_h_last,
        attributes.idx_valid_w_last,
        attributes.Ht_per_batch_index,
        attributes.output_logical_w,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*src_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderStreaming;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.named_compile_time_args = {
        {"reduction_mode", attributes.reduction_mode},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {
        kCbInput, kCbOutput, attributes.Ht, attributes.Wt_output, split.num_cores};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterStreaming;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Same common/per-core split as the interleaved factory; the kernels share the ABI.
    reader_desc.emplace_common_runtime_args({index_t.buffer(), src_t.buffer()});
    writer_desc.emplace_common_runtime_args({in_t.buffer(), output_tensor.buffer()});
    const uint32_t tile_w = in_t.tensor_spec().tile().get_width();
    const uint32_t tile_h = in_t.tensor_spec().tile().get_height();
    const auto page_map_vec = to_vector(attributes.page_map);
    uint32_t id = 0;
    for (const auto& [core, n] : scatter_assigned_cores(split, attributes.Wt_output)) {
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(n);
        reader_args.push_back(tile_w);
        reader_args.push_back(tile_h);
        reader_args.push_back(id);
        reader_args.append(page_map_vec);
        reader_desc.emplace_runtime_args(core, reader_args);
        writer_desc.emplace_runtime_args(core, {n, id});
        id++;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryRowMajor::create_descriptor(
    const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.index_tensor;
    const auto& src_t = tensor_args.src_tensor;

    auto* device = in_t.device();
    const auto split = split_scatter_work(*device, attributes.sub_core_grids, attributes.num_sticks);

    const uint32_t index_elem_size = index_t.element_size();
    const uint32_t src_elem_size = src_t.element_size();
    const uint32_t output_elem_size = output_tensor.element_size();
    // TensorAccessor's explicit page_size argument is the stride Buffer addressing uses to find a
    // page's bank offset, which is the buffer's ALIGNED page size, not the raw stick byte width --
    // passing the raw width addresses every page past the first at the wrong offset whenever the
    // stick isn't already a multiple of the device's alignment. Computed through
    // scatter_rm_stick_page_bytes() (not in_t.buffer()->aligned_page_size() directly) so this can
    // never drift from supported_by_codegen()'s own feasibility arithmetic over the same stick.
    const uint32_t input_page_bytes =
        static_cast<uint32_t>(scatter_rm_stick_page_bytes(in_t, attributes.input_stick_elems));
    const uint32_t output_page_bytes = static_cast<uint32_t>(
        scatter_output_aligned_page_size(in_t, attributes.output_mem_config, tensor_args.output_tensor));
    const uint32_t index_page_bytes = static_cast<uint32_t>(index_t.buffer()->aligned_page_size());
    const uint32_t src_page_bytes = static_cast<uint32_t>(src_t.buffer()->aligned_page_size());
    const uint32_t chunk_elems = scatter_rm_chunk_elems(
        scatter_usable_l1(in_t),
        scatter_rm_fixed_bytes(input_page_bytes, output_page_bytes, /*bf16_reduce=*/false),
        attributes.index_stick_elems,
        index_elem_size,
        src_elem_size);
    const uint32_t index_chunk_bytes = chunk_elems * index_elem_size;
    const uint32_t src_chunk_bytes = chunk_elems * src_elem_size;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_rm_cb(kCbOutput, output_tensor.dtype(), output_page_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbIndex, index_t.dtype(), index_chunk_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbSrc, src_t.dtype(), src_chunk_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbInput, in_t.dtype(), input_page_bytes, 1, split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbOutput,
        kCbIndex,
        kCbSrc,
        attributes.input_stick_elems,
        attributes.index_stick_elems,
        output_elem_size,
        index_elem_size,
        src_elem_size,
        index_page_bytes,
        src_page_bytes,
        chunk_elems,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*src_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderRm;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    // A ROW_MAJOR reduce routes to ScatterCodegenProgramFactoryBf16ReduceRowMajor instead
    // (select_program_factory), so this reader only ever runs with reduction_mode == kReductionNone,
    // which it static_asserts.
    reader_desc.named_compile_time_args = {
        {"reduction_mode", attributes.reduction_mode},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {kCbInput, kCbOutput, input_page_bytes, output_page_bytes};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterRm;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Buffer addresses are common (per-kernel) runtime args, so a program-cache hit rebinds four
    // slots in total rather than four per core. Per-core RT is the contiguous [start, n) stick range
    // (offset before count). Kernel ABI: reader common [index_addr, src_addr], per-core [start, n,
    // page_map...]; writer common [in_addr, out_addr], per-core [start, n].
    reader_desc.emplace_common_runtime_args({index_t.buffer(), src_t.buffer()});
    writer_desc.emplace_common_runtime_args({in_t.buffer(), output_tensor.buffer()});
    const auto page_map_vec = to_vector(attributes.page_map);
    uint32_t start = 0;
    for (const auto& [core, n] : scatter_assigned_cores(split, attributes.num_sticks)) {
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(start);
        reader_args.push_back(n);
        reader_args.append(page_map_vec);
        reader_desc.emplace_runtime_args(core, reader_args);
        writer_desc.emplace_runtime_args(core, {start, n});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryBf16ReduceRowMajor::create_descriptor(
    const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor) {
    const auto& in_t = tensor_args.input_tensor;
    const auto& index_t = tensor_args.index_tensor;
    const auto& src_t = tensor_args.src_tensor;

    auto* device = in_t.device();
    const auto split = split_scatter_work(*device, attributes.sub_core_grids, attributes.num_sticks);

    const uint32_t index_elem_size = index_t.element_size();
    const uint32_t src_elem_size = src_t.element_size();
    // See ScatterCodegenProgramFactoryRowMajor::create_descriptor: TensorAccessor's page_size argument
    // must be each buffer's ALIGNED page size, not the raw stick byte width, via the same
    // scatter_rm_stick_page_bytes() helper the feasibility gate uses.
    const uint32_t input_page_bytes =
        static_cast<uint32_t>(scatter_rm_stick_page_bytes(in_t, attributes.input_stick_elems));
    const uint32_t output_page_bytes = static_cast<uint32_t>(
        scatter_output_aligned_page_size(in_t, attributes.output_mem_config, tensor_args.output_tensor));
    const uint32_t index_page_bytes = static_cast<uint32_t>(index_t.buffer()->aligned_page_size());
    const uint32_t src_page_bytes = static_cast<uint32_t>(src_t.buffer()->aligned_page_size());
    const uint32_t fp32_temp_page_bytes = attributes.input_stick_elems * 4;
    const uint32_t chunk_elems = scatter_rm_chunk_elems(
        scatter_usable_l1(in_t),
        // The FP32 accumulator is L1-only scratch of exactly input_stick_elems*4 bytes, not a device
        // buffer page; 2x an aligned BF16 page upper-bounds it.
        scatter_rm_fixed_bytes(input_page_bytes, output_page_bytes, /*bf16_reduce=*/true),
        attributes.index_stick_elems,
        index_elem_size,
        src_elem_size);
    const uint32_t index_chunk_bytes = chunk_elems * index_elem_size;
    const uint32_t src_chunk_bytes = chunk_elems * src_elem_size;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_rm_cb(kCbOutput, output_tensor.dtype(), output_page_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbIndex, index_t.dtype(), index_chunk_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbSrc, src_t.dtype(), src_chunk_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbInput, in_t.dtype(), input_page_bytes, 1, split.core_range));
    desc.cbs.push_back(make_rm_cb(kCbFp32Temp, DataType::FLOAT32, fp32_temp_page_bytes, 1, split.core_range));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        kCbInput,
        kCbOutput,
        kCbIndex,
        kCbSrc,
        kCbFp32Temp,
        attributes.input_stick_elems,
        attributes.index_stick_elems,
        index_elem_size,
        src_elem_size,
        index_page_bytes,
        src_page_bytes,
        chunk_elems,
    };
    TensorAccessorArgs(*index_t.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*src_t.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderBf16ReduceRm;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.core_range;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.named_compile_time_args = {
        {"reduction_mode", attributes.reduction_mode},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct = {kCbInput, kCbOutput, input_page_bytes, output_page_bytes};
    TensorAccessorArgs(*in_t.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterRm;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.core_range;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    // Same common/per-core split as the plain ROW_MAJOR factory: the writer kernel is shared.
    reader_desc.emplace_common_runtime_args({index_t.buffer(), src_t.buffer()});
    writer_desc.emplace_common_runtime_args({in_t.buffer(), output_tensor.buffer()});
    const auto page_map_vec = to_vector(attributes.page_map);
    uint32_t start = 0;
    for (const auto& [core, n] : scatter_assigned_cores(split, attributes.num_sticks)) {
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(start);
        reader_args.push_back(n);
        reader_args.append(page_map_vec);
        reader_desc.emplace_runtime_args(core, reader_args);
        writer_desc.emplace_runtime_args(core, {start, n});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim

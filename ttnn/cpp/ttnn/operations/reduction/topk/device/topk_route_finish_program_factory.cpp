// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Program factory for topk_route_finish (fused TILE-source gather + tile assembly + dtype emit).
//
// Work unit = HALF of one output tile (a 16-row face-pair): unit(row_tile, kt, half) covers
// output rows [row_tile*32 + half*16, +16) x output cols [kt*32, +32), where row_tile is
// GLOBAL (all batches, tile-row-major) and kt indexes the k_rounded axis in 32-column tiles.
// Total units = total_tile_rows x K_t x 2, flat-listed unit-major and split across the full
// worker grid with a single cliff core (split_blocks_for_tilize, like topk_route_prep).
// Halves whose 16 rows are ALL tile-height padding (r >= logical R) are kept, not dropped:
// freshly allocated output pages hold garbage, and the contract promises zero-filled tile
// padding, so those units exist purely to write zeros.
//
// Per unit the gather load is SPLIT ACROSS BOTH RISCs: the reader (BRISC) owns unit rows
// [0, 8) and the writer (NCRISC) rows [8, 16). Each side reads its own <=8 RM u32
// index-stick segments into private scratch, zero-fills ITS row ranges of the two staging
// halves (one CB page each; the two row ranges touch disjoint 32 B face rows, so both RISCs
// may fill the same page concurrently — see topk_route_finish_gather_common.hpp), and
// gathers each selected bf16 element with a 64 B NoC read from the source tile's face-row
// into a rotating bounce slot. Gather reads are issued in 32-deep waves tagged with
// alternating NoC transaction ids (trids 1/2); each wave is retired with a per-trid barrier
// while the next wave's reads are in flight, so extraction overlaps flight instead of every
// wave paying a full-drain stall. The writer computes unit u's staging address from its own
// (never-blocking) read pointer BEFORE wait_front — safe because its own pop of unit u-2
// freed that page — gathers its rows, THEN wait_fronts (reader's rows done) and blasts each
// staged half contiguously into the output tile page at offset half * (page_size/2).
// Handshake = the two staging CBs themselves (2 pages each): the reader's push means
// "rows [0,8) + their zero fill done", the writer's pop means "page drained". The RISC
// split adds no blocking edge beyond that single producer/consumer pair (the full
// protocol + termination argument lives at the top of the writer kernel). No compute
// kernel: values are copied bit-exact from the TILE source (see the canonicalization note
// at the composite call site in topk.cpp).

#include "topk_route_finish_device_operation.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <limits>
#include <vector>

using namespace tt::constants;

namespace ttnn::operations::reduction::topk_route_finish::program {

namespace {

constexpr uint32_t reader_stick_cb_index = tt::CBIndex::c_0;   // reader-private index-stick staging
constexpr uint32_t reader_bounce_cb_index = tt::CBIndex::c_1;  // reader-private gather bounce slots
constexpr uint32_t writer_stick_cb_index = tt::CBIndex::c_2;   // writer-private index-stick staging
constexpr uint32_t writer_bounce_cb_index = tt::CBIndex::c_3;  // writer-private gather bounce slots
constexpr uint32_t values_cb_index = tt::CBIndex::c_16;        // staged value face-pairs, reader -> writer
constexpr uint32_t indices_cb_index = tt::CBIndex::c_17;       // staged index face-pairs, reader -> writer

// Kernel indices follow create_descriptor's push_back order.
constexpr uint32_t kReaderKernelIdx = 0;
constexpr uint32_t kWriterKernelIdx = 1;

// Per-core runtime arg slots. Must match the emplace_runtime_args order below.
namespace reader_arg {
constexpr uint32_t SRC_ADDR = 0;
constexpr uint32_t IDX_ADDR = 1;
constexpr uint32_t START_UNIT = 2;
constexpr uint32_t NUNITS = 3;
constexpr uint32_t LOGICAL_ROWS = 4;
constexpr uint32_t ROW_TILES_PER_BATCH = 5;
constexpr uint32_t K_ROUNDED = 6;
}  // namespace reader_arg
namespace writer_arg {
constexpr uint32_t VALUES_ADDR = 0;
constexpr uint32_t INDICES_ADDR = 1;
constexpr uint32_t START_UNIT = 2;
constexpr uint32_t NUNITS = 3;
constexpr uint32_t SRC_ADDR = 4;
constexpr uint32_t IDX_ADDR = 5;
constexpr uint32_t LOGICAL_ROWS = 6;
constexpr uint32_t ROW_TILES_PER_BATCH = 7;
constexpr uint32_t K_ROUNDED = 8;
}  // namespace writer_arg

// The sizing constants below must mirror topk_route_finish_gather_common.hpp — keep the
// two files in sync.
//
// Gather bounce buffers (one per RISC — each is that RISC's private scratch): 64 slots of
// 64 B. 64 B slot size (not the elements' 2 B) because Blackhole's NoC requires 64 B
// alignment on BOTH ends of a DRAM read (NOC_DRAM_READ_ALIGNMENT_BYTES); the gatherer
// aligns each source face-row read down to a 64 B boundary within the 2048 B tile page and
// extracts the bf16 at (byte & 63). The 64 slots hold the trid pipeline's two in-flight
// 32-read waves.
constexpr uint32_t bounce_slots = 64;
constexpr uint32_t bounce_slot_bytes = 64;

// Index-stick staging (one per RISC): one 128 B segment (32 u32 indices = one output tile
// width) per owned unit row, 8 rows per RISC (reader rows [0,8), writer rows [8,16)).
// kt*128 source offsets and 128 B row strides keep DRAM-read alignment.
constexpr uint32_t stick_segment_bytes = TILE_WIDTH * sizeof(uint32_t);
constexpr uint32_t stick_rows_per_risc = TILE_HEIGHT / 4;  // 8

struct FinishWorkSplit {
    uint32_t width_tiles = 0;          // logits W_p / 32
    uint32_t total_tile_rows = 0;      // logits padded volume / W_p / 32 (all batches)
    uint32_t row_tiles_per_batch = 0;  // logits R_p / 32
    uint32_t k_tiles = 0;              // div_up(k_rounded, 32)
    uint32_t k_rounded = 0;
    uint32_t total_units = 0;  // total_tile_rows * k_tiles * 2 (two face-pair halves per tile)
    bool index_is_u32 = false;
};

FinishWorkSplit compute_work_split(const Tensor& input, const Tensor& indices) {
    FinishWorkSplit split;
    const auto& padded = input.padded_shape();
    split.width_tiles = padded[-1] / TILE_WIDTH;
    split.total_tile_rows = (input.physical_volume() / padded[-1]) / TILE_HEIGHT;
    split.row_tiles_per_batch = padded[-2] / TILE_HEIGHT;
    split.k_rounded = indices.logical_shape()[-1];
    split.k_tiles = tt::div_up(split.k_rounded, TILE_WIDTH);
    split.total_units = split.total_tile_rows * split.k_tiles * 2;
    split.index_is_u32 = padded[-1] > std::numeric_limits<uint16_t>::max();
    return split;
}

struct FinishCoreArgs {
    CoreCoord core;
    uint32_t start_unit = 0;
    uint32_t nunits_this_core = 0;
};

// Per-core runtime layout. create_descriptor and override_runtime_arguments both call this so
// the arg slots cannot drift. The program hash pins K_t/W_t, page sizes, index dtype, and the
// core partition; logical rows are not in that hash.
struct FinishRuntimeLayout {
    FinishWorkSplit split;
    tt::tt_metal::CoreRangeSet all_cores;
    std::vector<FinishCoreArgs> cores;
    uint32_t logical_rows = 0;
};

FinishRuntimeLayout build_runtime_layout(const Tensor& input, const Tensor& indices) {
    FinishRuntimeLayout layout;
    layout.split = compute_work_split(input, indices);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto unit_split = ttnn::split_blocks_for_tilize(CoreCoord(grid.x, grid.y), layout.split.total_units);
    layout.all_cores = unit_split.all_cores;
    layout.logical_rows = input.logical_shape()[-2];

    // split_blocks_for_tilize(CoreCoord, ...) places core i at (i % grid.x, i / grid.x), cliff
    // last; enumerate in that exact order so the contiguous unit partition lines up.
    layout.cores.reserve(unit_split.ncores);
    uint32_t start_unit = 0;
    for (uint32_t i = 0; i < unit_split.ncores; ++i) {
        const bool is_cliff = unit_split.nblocks_per_core_cliff > 0 && i + 1 == unit_split.ncores;
        const uint32_t nunits_this_core = is_cliff ? unit_split.nblocks_per_core_cliff : unit_split.nblocks_per_core;
        layout.cores.push_back(FinishCoreArgs{
            .core = CoreCoord{i % grid.x, i / grid.x},
            .start_unit = start_unit,
            .nunits_this_core = nunits_this_core,
        });
        start_unit += nunits_this_core;
    }
    return layout;
}

tt::tt_metal::CBDescriptor make_cb(
    uint32_t total_size,
    const tt::tt_metal::CoreRangeSet& cores,
    uint32_t cb_index,
    tt::DataFormat format,
    uint32_t page_size) {
    return tt::tt_metal::CBDescriptor{
        .total_size = total_size,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = format,
            .page_size = page_size,
        }}},
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor TopkRouteFinishProgramFactory::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;
    const Tensor& values_out = std::get<0>(tensor_return_value);
    const Tensor& indices_out = std::get<1>(tensor_return_value);
    const auto layout = build_runtime_layout(input, indices);
    const auto& split = layout.split;
    const auto& all_cores = layout.all_cores;

    auto* input_buffer = input.buffer();
    auto* indices_buffer = indices.buffer();
    auto* values_buffer = values_out.buffer();
    auto* indices_out_buffer = indices_out.buffer();
    TT_FATAL(input_buffer != nullptr, "topk_route_finish logits input has no device buffer");
    TT_FATAL(indices_buffer != nullptr, "topk_route_finish indices input has no device buffer");
    TT_FATAL(values_buffer != nullptr, "topk_route_finish values output has no device buffer");
    TT_FATAL(indices_out_buffer != nullptr, "topk_route_finish indices output has no device buffer");

    const uint32_t value_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);  // 2048
    const uint32_t value_half_bytes = value_tile_bytes / 2;                      // one face-pair
    const uint32_t idx_elem_bytes = split.index_is_u32 ? 4 : 2;
    const uint32_t idx_half_bytes = (TILE_HW / 2) * idx_elem_bytes;  // 512 elements per face-pair
    const tt::DataFormat idx_format = split.index_is_u32 ? tt::DataFormat::UInt32 : tt::DataFormat::UInt16;

    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs.reserve(6);
    desc.kernels.reserve(2);

    // Per-RISC private scratch (a stick-stage + bounce pair for EACH data-movement RISC),
    // allocated as 1-page CBs (never pushed/popped, so the write pointer stays at the
    // base). CB bases are aligned to the DRAM alignment (64 B) by the program CB
    // allocator, which the 64 B bounce slots and 128 B stick rows rely on.
    constexpr uint32_t stick_cb_bytes = stick_rows_per_risc * stick_segment_bytes;
    for (const auto stick_cb_index : {reader_stick_cb_index, writer_stick_cb_index}) {
        desc.cbs.push_back(make_cb(stick_cb_bytes, all_cores, stick_cb_index, tt::DataFormat::UInt32, stick_cb_bytes));
    }

    for (const auto bounce_cb_index : {reader_bounce_cb_index, writer_bounce_cb_index}) {
        desc.cbs.push_back(make_cb(
            bounce_slots * bounce_slot_bytes,
            all_cores,
            bounce_cb_index,
            tt::DataFormat::Float16_b,
            bounce_slots * bounce_slot_bytes));
    }

    // Reader -> writer staging: one page per face-pair half, double-buffered. The page byte
    // layout is EXACTLY the output tile's face-pair range, so the writer issues one
    // contiguous write per page.
    desc.cbs.push_back(
        make_cb(2 * value_half_bytes, all_cores, values_cb_index, tt::DataFormat::Float16_b, value_half_bytes));
    desc.cbs.push_back(make_cb(2 * idx_half_bytes, all_cores, indices_cb_index, idx_format, idx_half_bytes));

    std::vector<uint32_t> reader_compile_args = {
        split.k_tiles,
        split.width_tiles,
        reader_stick_cb_index,
        reader_bounce_cb_index,
        values_cb_index,
        indices_cb_index,
        split.index_is_u32 ? 1u : 0u};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(reader_compile_args);
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_topk_route_finish_gather.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_args);
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_args = {
        split.k_tiles,
        split.width_tiles,
        values_cb_index,
        indices_cb_index,
        writer_stick_cb_index,
        writer_bounce_cb_index,
        value_half_bytes,
        idx_half_bytes,
        split.index_is_u32 ? 1u : 0u};
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(indices_out_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(writer_compile_args);
    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_topk_route_finish_tiles.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_args);
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    for (const auto& core_args : layout.cores) {
        const auto& core = core_args.core;
        reader_desc.emplace_runtime_args(
            core,
            {input_buffer,
             indices_buffer,
             core_args.start_unit,
             core_args.nunits_this_core,
             layout.logical_rows,
             split.row_tiles_per_batch,
             split.k_rounded});
        writer_desc.emplace_runtime_args(
            core,
            {values_buffer,
             indices_out_buffer,
             core_args.start_unit,
             core_args.nunits_this_core,
             input_buffer,
             indices_buffer,
             layout.logical_rows,
             split.row_tiles_per_batch,
             split.k_rounded});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

void TopkRouteFinishProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;
    const Tensor& values_out = std::get<0>(tensor_return_value);
    const Tensor& indices_out = std::get<1>(tensor_return_value);

    auto* input_buffer = input.buffer();
    auto* indices_buffer = indices.buffer();
    auto* values_buffer = values_out.buffer();
    auto* indices_out_buffer = indices_out.buffer();
    TT_FATAL(input_buffer != nullptr, "topk_route_finish logits input has no device buffer");
    TT_FATAL(indices_buffer != nullptr, "topk_route_finish indices input has no device buffer");
    TT_FATAL(values_buffer != nullptr, "topk_route_finish values output has no device buffer");
    TT_FATAL(indices_out_buffer != nullptr, "topk_route_finish indices output has no device buffer");

    const auto layout = build_runtime_layout(input, indices);
    const uint32_t input_address = input_buffer->address();
    const uint32_t indices_address = indices_buffer->address();
    const uint32_t values_address = values_buffer->address();
    const uint32_t indices_out_address = indices_out_buffer->address();

    for (const auto& core_args : layout.cores) {
        const auto& core = core_args.core;
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader_args[reader_arg::SRC_ADDR] = input_address;
        reader_args[reader_arg::IDX_ADDR] = indices_address;
        reader_args[reader_arg::START_UNIT] = core_args.start_unit;
        reader_args[reader_arg::NUNITS] = core_args.nunits_this_core;
        reader_args[reader_arg::LOGICAL_ROWS] = layout.logical_rows;
        reader_args[reader_arg::ROW_TILES_PER_BATCH] = layout.split.row_tiles_per_batch;
        reader_args[reader_arg::K_ROUNDED] = layout.split.k_rounded;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        writer_args[writer_arg::VALUES_ADDR] = values_address;
        writer_args[writer_arg::INDICES_ADDR] = indices_out_address;
        writer_args[writer_arg::START_UNIT] = core_args.start_unit;
        writer_args[writer_arg::NUNITS] = core_args.nunits_this_core;
        writer_args[writer_arg::SRC_ADDR] = input_address;
        writer_args[writer_arg::IDX_ADDR] = indices_address;
        writer_args[writer_arg::LOGICAL_ROWS] = layout.logical_rows;
        writer_args[writer_arg::ROW_TILES_PER_BATCH] = layout.split.row_tiles_per_batch;
        writer_args[writer_arg::K_ROUNDED] = layout.split.k_rounded;
    }
}

}  // namespace ttnn::operations::reduction::topk_route_finish::program

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

using namespace tt::constants;

namespace ttnn::operations::reduction::topk_route_finish::program {

namespace {

constexpr uint32_t reader_stick_cb_index = tt::CBIndex::c_0;   // reader-private index-stick staging
constexpr uint32_t reader_bounce_cb_index = tt::CBIndex::c_1;  // reader-private gather bounce slots
constexpr uint32_t writer_stick_cb_index = tt::CBIndex::c_2;   // writer-private index-stick staging
constexpr uint32_t writer_bounce_cb_index = tt::CBIndex::c_3;  // writer-private gather bounce slots
constexpr uint32_t values_cb_index = tt::CBIndex::c_16;        // staged value face-pairs, reader -> writer
constexpr uint32_t indices_cb_index = tt::CBIndex::c_17;       // staged index face-pairs, reader -> writer

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

// (Re)computes every runtime arg from the tensors. Called by create() and, on cache hits, by
// override_runtime_arguments(); the program hash pins every structural input (core partition,
// K_t/W_t, page sizes, index dtype), so only addresses and the logical-R/k_rounded clamps can
// differ here.
void set_runtime_args(
    tt::tt_metal::Program& program,
    const TopkRouteFinishSharedVariables& shared,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& values_out,
    const Tensor& indices_out) {
    const auto split = compute_work_split(input, indices);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto unit_split = ttnn::split_blocks_for_tilize(CoreCoord(grid.x, grid.y), split.total_units);

    const uint32_t logical_rows = input.logical_shape()[-2];

    uint32_t start_unit = 0;
    for (uint32_t i = 0; i < shared.cores.size(); ++i) {
        const CoreCoord& core = shared.cores[i];
        const bool is_cliff = unit_split.nblocks_per_core_cliff > 0 && i + 1 == shared.cores.size();
        const uint32_t nunits_this_core = is_cliff ? unit_split.nblocks_per_core_cliff : unit_split.nblocks_per_core;

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            core,
            {input.buffer()->address(),    // src_addr (TILE bf16 logits)
             indices.buffer()->address(),  // idx_addr (RM u32 sticks)
             start_unit,
             nunits_this_core,
             logical_rows,               // R (per-unit valid-row clamp)
             split.row_tiles_per_batch,  // R_p / 32 (unit -> batch decomposition)
             split.k_rounded});          // per-unit valid-column clamp

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.writer_kernel_id,
            core,
            {values_out.buffer()->address(),   // dst values
             indices_out.buffer()->address(),  // dst indices
             start_unit,
             nunits_this_core,
             input.buffer()->address(),    // src_addr (TILE bf16 logits, writer's gather rows)
             indices.buffer()->address(),  // idx_addr (RM u32 sticks, writer's gather rows)
             logical_rows,                 // R (per-unit valid-row clamp)
             split.row_tiles_per_batch,    // R_p / 32 (unit -> batch decomposition)
             split.k_rounded});            // per-unit valid-column clamp

        start_unit += nunits_this_core;
    }
}

}  // namespace

TopkRouteFinishProgramFactory::cached_program_t TopkRouteFinishProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;
    const Tensor& values_out = std::get<0>(tensor_return_value);
    const Tensor& indices_out = std::get<1>(tensor_return_value);

    auto program = tt::tt_metal::CreateProgram();

    const auto split = compute_work_split(input, indices);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto unit_split = ttnn::split_blocks_for_tilize(CoreCoord(grid.x, grid.y), split.total_units);
    const auto& all_cores = unit_split.all_cores;

    const uint32_t value_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);  // 2048
    const uint32_t value_half_bytes = value_tile_bytes / 2;                      // one face-pair
    const uint32_t idx_elem_bytes = split.index_is_u32 ? 4 : 2;
    const uint32_t idx_half_bytes = (TILE_HW / 2) * idx_elem_bytes;  // 512 elements per face-pair
    const tt::DataFormat idx_format = split.index_is_u32 ? tt::DataFormat::UInt32 : tt::DataFormat::UInt16;

    // Per-RISC private scratch (a stick-stage + bounce pair for EACH data-movement RISC),
    // allocated as 1-page CBs (never pushed/popped, so the write pointer stays at the
    // base). CB bases are aligned to the DRAM alignment (64 B) by the program CB
    // allocator, which the 64 B bounce slots and 128 B stick rows rely on.
    constexpr uint32_t stick_cb_bytes = stick_rows_per_risc * stick_segment_bytes;
    for (const auto stick_cb_index : {reader_stick_cb_index, writer_stick_cb_index}) {
        const auto stick_stage_cb_config =
            tt::tt_metal::CircularBufferConfig(stick_cb_bytes, {{stick_cb_index, tt::DataFormat::UInt32}})
                .set_page_size(stick_cb_index, stick_cb_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, stick_stage_cb_config);
    }

    for (const auto bounce_cb_index : {reader_bounce_cb_index, writer_bounce_cb_index}) {
        const auto bounce_cb_config =
            tt::tt_metal::CircularBufferConfig(
                bounce_slots * bounce_slot_bytes, {{bounce_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(bounce_cb_index, bounce_slots * bounce_slot_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, bounce_cb_config);
    }

    // Reader -> writer staging: one page per face-pair half, double-buffered. The page byte
    // layout is EXACTLY the output tile's face-pair range, so the writer issues one
    // contiguous write per page.
    const auto values_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * value_half_bytes, {{values_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(values_cb_index, value_half_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, values_cb_config);

    const auto indices_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * idx_half_bytes, {{indices_cb_index, idx_format}})
            .set_page_size(indices_cb_index, idx_half_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, indices_cb_config);

    std::vector<uint32_t> reader_compile_args = {
        split.k_tiles,
        split.width_tiles,
        reader_stick_cb_index,
        reader_bounce_cb_index,
        values_cb_index,
        indices_cb_index,
        split.index_is_u32 ? 1u : 0u};
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*indices.buffer()).append_to(reader_compile_args);
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_topk_route_finish_gather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

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
    tt::tt_metal::TensorAccessorArgs(*values_out.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*indices_out.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*indices.buffer()).append_to(writer_compile_args);
    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_topk_route_finish_tiles.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // split_blocks_for_tilize(CoreCoord, ...) places core i at (i % grid.x, i / grid.x), cliff
    // last; enumerate in that exact order so the contiguous unit partition lines up.
    std::vector<CoreCoord> cores;
    cores.reserve(unit_split.ncores);
    for (uint32_t i = 0; i < unit_split.ncores; ++i) {
        cores.push_back(CoreCoord{i % grid.x, i / grid.x});
    }

    TopkRouteFinishSharedVariables shared{
        .reader_kernel_id = reader_kernel, .writer_kernel_id = writer_kernel, .cores = std::move(cores)};
    set_runtime_args(program, shared, input, indices, values_out, indices_out);

    return cached_program_t{std::move(program), std::move(shared)};
}

void TopkRouteFinishProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    set_runtime_args(
        cached_program.program,
        cached_program.shared_variables,
        tensor_args.input_tensor,
        tensor_args.indices_tensor,
        std::get<0>(tensor_return_value),
        std::get<1>(tensor_return_value));
}

}  // namespace ttnn::operations::reduction::topk_route_finish::program

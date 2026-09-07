// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for rms_norm_ttnn (NCRISC, NoC0).
//
// Per core, per row-block, per width chunk it stages:
//   * TILE build      : whole x tiles         -> cb_input_tiles
//                       whole residual tiles  -> cb_residual_tiles   (HAS_RESIDUAL)
//   * ROW_MAJOR build : padded x sticks       -> cb_input_sticks      (compute tilizes)
//                       padded residual sticks-> cb_residual_sticks   (HAS_RESIDUAL)
// plus, once at boot, the reduce scaler; plus the per-channel operands
// (`weight` -> cb_gamma_*, `bias` -> cb_bias_*), staged once per core in the
// RESIDENT regimes and once per width chunk in STREAM.
//
// Loop nest mirrors the compute kernel exactly:
//     for blk in blocks: for pass in {A} or {A,B}: for c in chunks
// Pass B re-reads the activations ONLY in the STREAM regime (X_RESIDENT == 0).
// Both resident regimes hold the pass-B operand across the two passes, so x and
// the residual are read once: RESIDENT because the row IS one chunk, ROW_RESIDENT
// (Lamp L5 / descriptor D14) because the whole tile-row is held while only the
// derived CBs are chunked.  The per-channel operands follow X_RESIDENT the same
// way -- staged once per core for every chunk of the row, instead of re-read per
// pass-B chunk of every row-block, which on a prefill profile is as many DRAM
// bytes as x itself.
//
// A1 -- THE RESIDUAL.  It carries the input's exact geometry by contract, so it
// rides the SAME three mechanisms on the SAME schedule and needs no new one:
//   * TILE interleaved: both streams are issued inside ONE width-tile loop and
//     covered by the ONE shared noc_async_read_barrier(), because that barrier
//     fences NoC-0 globally.  Two streams therefore cost ONE barrier per (block,
//     chunk), not two.
//   * ROW_MAJOR: each stream costs its own barrier, because
//     read_sticks_for_tilize owns its reserve/push at TILE granularity.
//   * native (a resident shard): ZERO-COPY.  The residual carries the identical
//     shard spec, so it is ALREADY in this core's L1; reading it through a
//     TensorAccessor would re-fetch resident bytes and add a redundant arena
//     copy.  The early return is gated PER STREAM (NATIVE_IN elides x's read,
//     NATIVE_RESIDUAL elides the residual's) so a future divergence between the
//     two flags fails loudly instead of silently computing norm(x) for
//     norm(x + r).
//   * BAND: staged out of the residual's OWN resident L1 shard, at the same
//     global tile frame x uses.
//
// A2 -- THE BIAS.  A per-channel operand read by the same BroadcastDim::Row
// consumer as the weight, so it takes the same staging path at its OWN dtype
// (`BIAS_ELEM_BYTES` / `BIAS_TRIM` are separate CT args -- the two operands share
// a LAYOUT but not a FORMAT, and a shared trim constant would truncate one).
//
// A10 -- THE BLOCKED PER-CHANNEL FORM.  A ROW_MAJOR operand arrives either flat
// (1,1,1,W) or blocked (Wt,32).  The two are byte-identical once staged, so only
// this kernel can tell them apart; see stage_per_channel_chunk.
//
// Helper-usage notes
// ------------------
// * scaler CB          -> dataflow_kernel_lib::prepare_[partial_]reduce_scalers
//                         (ReduceTile datapath) or prepare_reduce_mask
//                         (AccumulateViaAdd datapath), pool-type-aware
//                         overloads (PoolType::SUM, ReduceDim::REDUCE_ROW).
// * ROW_MAJOR staging  -> dataflow_kernel_lib::read_sticks_for_tilize at TILE
//                         granularity, which is exactly the contract of
//                         compute_kernel_lib::tilize<WT_CHUNK>(rows).
// * TILE staging + per-channel reads are TensorAccessor + noc_async_read[_tile]:
//   the dataflow tilize helper covers neither whole-tile interleaved reads nor
//   the per-channel slot.  It also cannot express the BLOCKED per-channel form:
//   it maps page p to tile ROW p, and that form needs page p at tile COLUMN p of
//   one staged stick -- a structural property of the helper, not a parameter.
//
// The BAND scheme (a ROW_MAJOR shard cutting the WIDTH axis): such a shard's page
// is a row SEGMENT, so no accessor read can reach a row.  Instead each core stages
// the band it already holds out of its OWN L1, and joins the unchanged cross-core
// combine: sum(t^2) over a row is the sum over the bands however the bands are
// cut.  See _plan_band in rms_norm_ttnn_program_descriptor.py.
//
// One raw-API addition beyond the design's table: a ONE-TIME zero of the whole
// cb_input_sticks (and cb_residual_sticks) ring at boot, via
// noc.async_write_zeros (the device zero API), gated on STAGE_ZERO.  Reason (R3):
// the L1 pad lanes of a staged ROW_MAJOR row are never written by a stick read,
// so whatever L1 garbage was there survives into the reduce.  The partial scaler
// multiplies pad lanes by zero, and inf*0 / nan*0 = NaN would poison the whole
// row.  Zeroing the rings once establishes the invariant "every pad byte is
// either zero or real tensor data" (later reads only ever overwrite with tensor
// values), so no per-block zeroing is needed.  With a residual the invariant has
// to cover BOTH rings: `x_pad + r_pad` reaches cb_x_sum, and the reduce's mask can
// only rescue it if it is finite.  H-tail rows need no zeroing: a padding row's
// reduction and output are confined to that row and the writer never writes it.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
// PERMANENT per-stage device-profiler instrumentation (never remove; free when
// the profiler is off -- see the header's durability contract).
// ---- TEMPORARY ABLATION SWITCHES (/perf-measure cumulative peel) -----------
// Uncomment to strip a stage's NoC PAYLOAD while keeping every CB handshake,
// barrier and trip count.  Perf measurement only -- the op is WRONG with any of
// these on.  They stay commented in the committed tree.
// #define RMS_ABLATE_READ_X
#define RMS_ABLATE_PER_CHANNEL
#include "perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_gamma_sticks = 5;
constexpr uint32_t cb_gamma_tiles = 6;
// Perf 3 / D27: the cross-core combine's one-hot permutation bank (bf16).  Synthesized
// here, consumed by the compute kernel's two matmul permutes, never popped.
constexpr uint32_t cb_bank = 14;
// A1 / A2 -- new slots, allocated only under their HAS_* flag.
constexpr uint32_t cb_residual_sticks = 19;
constexpr uint32_t cb_residual_tiles = 20;
constexpr uint32_t cb_bias_sticks = 22;
constexpr uint32_t cb_bias_tiles = 23;
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
// bf16 1.0 == 0x3F80 (sign 0, exponent 127, mantissa 0) -- EXACT, so the permutation
// matmuls multiply by a true unit and the bank costs nothing in accuracy.
constexpr uint16_t BF16_ONE = 0x3F80;
}  // namespace

// ===========================================================================
// A2 + A10 -- ONE staging implementation for BOTH per-channel operands.
// ===========================================================================
// `weight` and `bias` are read by the SAME BroadcastDim::Row consumer over the
// same channel axis, so they stage identically; only their dtype (hence
// ELEM_BYTES / TILE_BYTES / TRIM) and their CB pair differ, and both are template
// parameters.  Writing it once is what keeps the bias from drifting away from the
// weight's measured read granularity.
//
// THE THREE FORMS, and why only this kernel can tell them apart:
//
//   ROW_MAJOR flat (1,1,1,W)   ONE wide read of the chunk's bytes at byte offset
//                              first_wt * 32 * elem.  read_sticks_for_tilize
//                              stages it as row 0 of the chunk's tile-column
//                              block, which is the only row the consumer reads.
//   ROW_MAJOR blocked (Wt,32)  WT_CHUNK per-PAGE reads of 32 * elem bytes, page
//                              first_wt + w landing at byte offset w * 32 * elem
//                              of the SAME staged stick.  Byte-identical staging
//                              to the flat form -- which is exactly why nothing
//                              downstream of this function knows the difference.
//                              It cannot be a wide read: the blocked form's Wt
//                              pages are interleaved across DRAM banks, so the
//                              flat form's single transaction would read the
//                              wrong bytes.
//   TILE                       one read per tile at the TRIM granularity (D23).
//
// TILE-COLUMN ALIGNMENT is load-bearing, not incidental: a DRAM read whose source
// offset is not 64-byte aligned is silently TRUNCATED down to the alignment
// (measured: bands 1..3 of an 8-element shard all received weight[0..8)).  Every
// offset here is a multiple of 32 * elem -- a tile column -- for every dtype,
// which is why the BAND scheme stages into the tensor's GLOBAL tile frame rather
// than at the band's own byte offset.
template <
    uint32_t CB_STICKS,
    uint32_t CB_TILES,
    uint32_t WT,
    uint32_t WT_CHUNK,
    uint32_t ELEM_BYTES,
    uint32_t W_ELEMS,
    uint32_t TRIM,
    bool IS_RM,
    bool BLOCKED,
    bool NARROW,
    typename Acc>
FORCE_INLINE void stage_per_channel_chunk(const Acc& acc, uint32_t first_wt) {
    constexpr uint32_t TILE_COL_BYTES = TILE_DIM * ELEM_BYTES;
    constexpr uint32_t CHUNK_BYTES = WT_CHUNK * TILE_COL_BYTES;
    if constexpr (IS_RM && NARROW) {
        // D30 -- THE NARROW STAGING RING.  One tile COLUMN per page, so the ring is
        // a knob-derived DEPTH instead of a WT_CHUNK-wide block: the compute side
        // consumes it as `tilize<1, sticks, tiles>(WT_CHUNK)` -- WT_CHUNK one-tile
        // blocks instead of one WT_CHUNK-wide block -- and the staged tiles are
        // IDENTICAL, because tile column j of a wide block and block j of a
        // one-wide walk are the same 32 elements laid out the same way.
        //
        // WHY IT EXISTS.  A per-channel operand is ONE stick, but a
        // `tilize<WT_CHUNK>` staging ring has to reserve 32 rows' worth of pages to
        // carry it -- WT_CHUNK whole tiles of L1 for 32 * WT_CHUNK real elements.
        // On the widest band in the suite (a 8192-element fp32 row block-sharded
        // over 88 cores, 25 tile columns per band) that is 100 kB per operand, and
        // with two per-channel operands plus a residual it is what pushed the CB
        // region 62 kB past L1.  It costs WT_CHUNK LLK block calls instead of one,
        // paid once per core in the resident regimes, which is why the descriptor
        // takes it only when the budget asks (see the band search).
        //
        // THE TWO FORMS COLLAPSE HERE.  Flat and blocked differ by exactly one
        // expression -- which page the 32 elements live on and at what offset -- so
        // the narrow path needs no separate branch for them.
        for (uint32_t w = 0; w < WT_CHUNK; ++w) {
            const uint32_t wt = first_wt + w;
            // A RAGGED width shard's last core owns fewer real tile columns than
            // WT_CHUNK; clamp so the read stays inside the tensor (the product lands
            // in the output's pad region and is never read back).
            const uint32_t real_wt = (wt < WT) ? wt : (WT - 1);
            cb_reserve_back(CB_STICKS, 1);
            const uint32_t dst = get_write_ptr(CB_STICKS);
            if constexpr (BLOCKED) {
                noc_async_read(acc.get_noc_addr(real_wt, 0), dst, TILE_COL_BYTES);
            } else {
                // Tile-column offsets are multiples of 32 * elem -- 128 B at fp32,
                // 64 B at bf16 -- so every source offset is 64-byte DRAM aligned.
                // That is load-bearing: an unaligned source offset is silently
                // TRUNCATED down to the alignment.
                noc_async_read(acc.get_noc_addr(0, real_wt * TILE_COL_BYTES), dst, TILE_COL_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(CB_STICKS, 1);
        }
    } else if constexpr (IS_RM) {
        if constexpr (BLOCKED) {
            cb_reserve_back(CB_STICKS, WT_CHUNK);
            const uint32_t l1_base = get_write_ptr(CB_STICKS);
            for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                // A RAGGED width shard's last core owns fewer real tile columns
                // than WT_CHUNK; clamp so the read stays inside the tensor (the
                // product lands in the output's pad region and is never read back).
                const uint32_t wt = first_wt + w;
                const uint32_t page = (wt < WT) ? wt : (WT - 1);
                noc_async_read(acc.get_noc_addr(page, 0), l1_base + w * TILE_COL_BYTES, TILE_COL_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(CB_STICKS, WT_CHUNK);
        } else {
            // A single stick; row 0 of the staged tile-row is the only row
            // BroadcastDim::Row reads.
            const uint32_t off = first_wt * TILE_COL_BYTES;
            const uint32_t total = W_ELEMS * ELEM_BYTES;
            const uint32_t remaining = (off < total) ? (total - off) : 0;
            const uint32_t row_bytes = (remaining < CHUNK_BYTES) ? remaining : CHUNK_BYTES;
            uint32_t pushed = 0;
            if (row_bytes != 0) {
                dataflow_kernel_lib::read_sticks_for_tilize<CB_STICKS>(
                    acc,
                    /*total_num_rows=*/1,
                    row_bytes,
                    /*start_page=*/0,
                    /*byte_offset_within_page=*/off);
                // The helper pushes ceil(row_bytes / tile-column) pages, NOT
                // WT_CHUNK (tilize_helpers_dataflow.inl width_in_tiles).
                pushed = (row_bytes + TILE_COL_BYTES - 1) / TILE_COL_BYTES;
            }
            if (pushed < WT_CHUNK) {
                // A RAGGED width shard's last core owns fewer real tile columns
                // than WT_CHUNK.  tilize<WT_CHUNK> waits for the full block, so
                // top the push up: those pages tilize into the PAD tile columns,
                // whose product lands in the output's pad region.
                cb_reserve_back(CB_STICKS, WT_CHUNK - pushed);
                cb_push_back(CB_STICKS, WT_CHUNK - pushed);
            }
        }
    } else {
        const uint32_t tile_bytes = get_tile_size(CB_TILES);
        cb_reserve_back(CB_TILES, WT_CHUNK);
        uint32_t l1_addr = get_write_ptr(CB_TILES);
        for (uint32_t w = 0; w < WT_CHUNK; ++w) {
            const uint32_t wt = first_wt + w;
            const uint32_t tile_id = (wt < WT) ? wt : (WT - 1);
            // D23: fetch only the part of the tile pass B's BroadcastDim::Row
            // consumer reads.  A per-channel operand is a (1,1,1,W) vector, so 31
            // of the tile's 32 rows are PADDING and row 0 lives in the top
            // row-group of faces 0 and 1.  Everything the trim does not fetch
            // stays whatever was in the CB -- which is exactly why the trim is
            // only as wide as the faces the consumer provably reads (measured:
            // rows 1..31 seeded 1e5x wrong left the output BIT-IDENTICAL;
            // corrupting row 0 instead collapsed pcc).
#ifdef RMS_ABLATE_PER_CHANNEL
            (void)tile_id;
#else
            if constexpr (TRIM == 2) {
                // Two face-rows.  The face offset tile_bytes/4 is 64-byte DRAM
                // aligned for every LINEAR tiled format -- the descriptor has
                // already refused this granularity for block-float, whose
                // 272-byte face is not.
                constexpr uint32_t ROW_BYTES = TILE_DIM * ELEM_BYTES;
                const uint64_t base = get_noc_addr(tile_id, acc);
                const uint32_t face = tile_bytes / 4;
                noc_async_read(base, l1_addr, ROW_BYTES);
                noc_async_read(base + face, l1_addr + face, ROW_BYTES);
            } else if constexpr (TRIM == 1) {
                // Half the page from offset 0 == faces 0 and 1 in EVERY tiled
                // format, block-float included; needs no face-stride alignment.
                noc_async_read(get_noc_addr(tile_id, acc), l1_addr, tile_bytes / 2);
            } else {
                noc_async_read_tile(tile_id, acc, l1_addr);
            }
#endif
            l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(CB_TILES, WT_CHUNK);
    }
}

// ---- native activations: publish a resident shard's pages, once -------------
// The shard IS the per-core block, so the only thing to do is make its pages
// visible to the compute kernel -- there is no NoC read for it at all.  A RAGGED
// width shard (Wt not a multiple of the shard's tile width) ends each of its
// tile-rows in whole PAD tiles whose L1 content is undefined; zero them once so
// they contribute exactly 0 to sum(t^2), which is the same pad-lane invariant the
// ROW_MAJOR staging rings get at boot.  A1: the residual's shard gets the
// identical treatment -- same geometry, same pad tiles, same invariant -- which is
// the whole reason this is a function with the CB as a template parameter rather
// than the seed's inline block.
template <uint32_t CB, uint32_t WT_CHUNK, uint32_t IN_SHARD_PAGES>
FORCE_INLINE void publish_native_shard(uint32_t w_real, uint32_t tile_bytes) {
    if (w_real < WT_CHUNK) {
        const uint32_t pad_tiles = WT_CHUNK - w_real;
        Noc noc;
        DataflowBuffer dfb(CB);
        for (uint32_t r = 0; r * WT_CHUNK < IN_SHARD_PAGES; ++r) {
            noc.async_write_zeros(dfb, pad_tiles * tile_bytes, {.offset_bytes = (r * WT_CHUNK + w_real) * tile_bytes});
        }
        noc.write_zeros_l1_barrier();
    }
    cb_reserve_back(CB, IN_SHARD_PAGES);
    cb_push_back(CB, IN_SHARD_PAGES);
}

// ---- D33: the RAGGED TAIL of a ROW_MAJOR width chunk ------------------------
// `read_sticks_for_tilize` derives its L1 stride from `row_bytes`, so on a tail
// chunk that is narrower than WT_CHUNK it would pack the sticks at the REAL width
// while `tilize<WT_CHUNK>` reads them back at the PADDED one.  The tail therefore
// stages RAW -- one read per stick into a WT_CHUNK-wide row -- which is exactly the
// shape of the BAND scheme's `stage_band`, with the trailing pad lanes zeroed so
// they contribute exactly 0 to sum(t^2).
//
// RAW-API JUSTIFICATION (the helper cannot express this): the destination stride is
// the only thing that differs from the helper's TILE mode, and `row_bytes` is the
// helper's ONE source for both the bytes read and the stride written.  A helper that
// took them separately would close the gap; until it does, this is the same
// nine-line strided read the band already runs.
template <uint32_t CB, uint32_t WT_CHUNK, uint32_t CHUNK_ROW_BYTES, uint32_t REAL_ROW_BYTES, typename Acc>
FORCE_INLINE void stage_ragged_tail_sticks(const Acc& acc, uint32_t sticks, uint32_t stick_start, uint32_t byte_off) {
    cb_reserve_back(CB, WT_CHUNK);
    {
        // ONE zero call, not one per stick: everything from the first stick's pad
        // lanes to the end of the reserved region covers every stick's pad lanes, and
        // the real lanes it also touches are overwritten by the reads below -- which
        // are issued only after the zero's own barrier, as the zero API requires.
        Noc noc;
        DataflowBuffer dfb(CB);
        noc.async_write_zeros(dfb, WT_CHUNK * get_tile_size(CB) - REAL_ROW_BYTES, {.offset_bytes = REAL_ROW_BYTES});
        noc.write_zeros_l1_barrier();
    }
    const uint32_t dst = get_write_ptr(CB);
    for (uint32_t i = 0; i < sticks; ++i) {
        noc_async_read(acc.get_noc_addr(stick_start + i, byte_off), dst + i * CHUNK_ROW_BYTES, REAL_ROW_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(CB, WT_CHUNK);
}

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_ttnn_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(3);
    // Refinement 3 / lever 3: index 4 packs BLOCK_ROWS (low half) with the NoC
    // TRANSACTION UNIT minus one (high half) -- see `_pack_txn_rows` in the
    // descriptor.  TXN_ROWS is always a DIVISOR of BLOCK_ROWS, which is what makes a
    // multi-tile-row reserve straddle-free: every block-scoped ring is
    // `depth * BLOCK_ROWS * WT_CHUNK` pages and a group starts block-aligned.  At the
    // default TXN_ROWS == 1 the word IS `block_rows` and this file is the seed's.
    constexpr uint32_t BLOCK_ROWS_CT = get_compile_time_arg_val(4);
    constexpr uint32_t BLOCK_ROWS = BLOCK_ROWS_CT & 0xFFFFu;
    constexpr uint32_t TXN_ROWS = (BLOCK_ROWS_CT >> 16) + 1;
    static_assert(TXN_ROWS >= 1 && BLOCK_ROWS % TXN_ROWS == 0, "rms_norm_ttnn: TXN_ROWS must divide BLOCK_ROWS");
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(6);
    // The two per-channel operands SHARE a layout by contract, so one flag covers
    // both readers.
    constexpr uint32_t PER_CHANNEL_IS_RM = get_compile_time_arg_val(7);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(8);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(9);
    // Total ROW_MAJOR sticks in the tensor.  The per-core stick range comes in as
    // a runtime extent (stick_base / stick_count), which is what the BAND scheme
    // needs (its rows do not start on a tile boundary); R_RM is kept as the
    // whole-tensor figure the CT-arg contract documents.
    [[maybe_unused]] constexpr uint32_t R_RM = get_compile_time_arg_val(10);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(11);
    constexpr uint32_t REDUCE_ACC_VIA_ADD = get_compile_time_arg_val(12);
    // NATIVE_IN == 1 means cb_input_tiles is BACKED ON THE INPUT SHARD
    // (ttnn.cb_descriptor_from_sharded_tensor).  x is already resident in this
    // core's L1, so there is no NoC read for it at all -- the reader only
    // PUBLISHES the pages once so cb_wait_front can see them.
    constexpr uint32_t NATIVE_IN = get_compile_time_arg_val(13);
    constexpr uint32_t IN_SHARD_PAGES = get_compile_time_arg_val(14);
    // BAND == 1 means this core stages x out of its OWN ROW_MAJOR shard -- an RM
    // shard that cuts the WIDTH axis, whose page is a row SEGMENT, so the accessor
    // cannot reach a row.  The core's shard IS its band (every stick it owns x
    // `shard_w` elements) at x_addr + local_stick * SHARD_ROW_BYTES, and the
    // cross-core combine sums the group's per-row partials elementwise, so the
    // band need not start or end on a tile column.
    constexpr uint32_t BAND = get_compile_time_arg_val(15);
    constexpr uint32_t SHARD_ROW_BYTES = get_compile_time_arg_val(16);
    // Whether the RM staging rings must be zeroed at boot (some staged stick is
    // narrower than the ring's padded row).  On the whole-row schemes this is
    // PARTIAL_W != 0; on the BAND scheme it is a band that does not fill its tile
    // columns, and there it REPLACES the reduce mask entirely.
    constexpr uint32_t STAGE_ZERO = get_compile_time_arg_val(17);
    // Lamp L5 (descriptor D14): the pass-B operand (and the per-channel operands)
    // are held across both passes.  An EXPLICIT flag rather than
    // `NUM_W_CHUNKS == 1`, which is what gives the op its third regime --
    // ROW_RESIDENT: resident hold with the DERIVED CBs chunked, i.e. ONE pass over
    // the activations here instead of two.
    constexpr uint32_t X_RES = get_compile_time_arg_val(18);
    // Perf 2 (descriptor D23): TILE per-channel read granularity.  0 whole tile /
    // 1 half page / 2 face-rows.  The descriptor owns the choice (it knows the
    // operand's tile format and whether a face offset is 64-byte DRAM aligned);
    // the kernel only spells the reads.
    constexpr uint32_t GAMMA_TRIM = get_compile_time_arg_val(19);
    // Perf 3 (descriptor D27): pages of the one-hot permutation bank this core
    // must synthesize into cb_bank -- BLOCK_ROWS on the cross-core width combine,
    // 0 everywhere else (which elides the whole `reader_bank_boot` zone).
    constexpr uint32_t BANK_PAGES = get_compile_time_arg_val(20);
    // ---- A1 / A2 / A10: appended, so an operand-free build is the seed's -------
    constexpr uint32_t HAS_BIAS = get_compile_time_arg_val(21);
    constexpr uint32_t BIAS_ELEM_BYTES = get_compile_time_arg_val(22);
    constexpr uint32_t BIAS_TRIM = get_compile_time_arg_val(23);
    constexpr uint32_t HAS_RESIDUAL = get_compile_time_arg_val(24);
    constexpr uint32_t NATIVE_RESIDUAL = get_compile_time_arg_val(25);
    constexpr uint32_t GAMMA_BLOCKED = get_compile_time_arg_val(26);
    constexpr uint32_t BIAS_BLOCKED = get_compile_time_arg_val(27);
    // D30: stage a ROW_MAJOR per-channel operand one tile COLUMN per page.
    constexpr uint32_t NARROW_PC_STAGE = get_compile_time_arg_val(28);
    // D33 -- THE RAGGED (PADDED) WIDTH CHUNK.  `WT_PAD` is how many of the LAST
    // chunk's WT_CHUNK width tiles this core does not own: the descriptor's
    // `_width_chunk` takes the coarsest BALANCED chunk at a prime Wt (127 -> 4 x 32)
    // instead of collapsing to the only divisor (1), and pads the tail out so that
    // every CB ring, every helper block width and every batched NoC group stays
    // UNIFORM.  0 on every divisor build, which elides all of this.
    //
    // THE READER OWNS THE PAD'S INVARIANT.  Pad tiles are never read from the tensor
    // -- their tile ids are outside the row -- and pass A sums x^2 over the whole
    // padded chunk, so they must be exactly ZERO, not merely finite.  Zeroing is the
    // device zero API on the reserved pages, the same mechanism (and the same
    // reason) as `publish_native_shard`'s ragged-shard pad and the RM rings' boot
    // zero.  Pass B's product for those columns lands in the writer's skipped
    // region and is never read back.
    constexpr uint32_t WT_PAD = get_compile_time_arg_val(29);
    constexpr bool HAS_WPAD = (WT_PAD != 0);
    constexpr uint32_t WT_REAL_TAIL = WT_CHUNK - WT_PAD;
    static_assert(WT_PAD < WT_CHUNK, "rms_norm_ttnn: a width chunk that is ALL pad is not a chunk");
    static_assert(!HAS_WPAD || PARTIAL_W == 0, "rms_norm_ttnn: a ragged width chunk requires a tile-aligned width");
    static_assert(!HAS_WPAD || BLOCK_ROWS == 1, "rms_norm_ttnn: a ragged width chunk holds ONE tile-row per block");
    static_assert(!HAS_WPAD || NUM_W_CHUNKS > 1, "rms_norm_ttnn: a one-chunk width is never padded");
    static_assert(!HAS_WPAD || BAND == 0, "rms_norm_ttnn: the BAND scheme's width is shard-derived, never chunked");
    static_assert(!HAS_WPAD || NATIVE_IN == 0, "rms_norm_ttnn: a zero-copy x is resident, never chunked");
    // FOUR accessor arg blocks, chained at compile time and ALWAYS declared --
    // never inside an `if constexpr`, or an absent operand would shift every
    // later block's offset.  The descriptor emits the NULL form for an absent one.
    constexpr auto x_args = TensorAccessorArgs<30>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();

    constexpr bool NATIVE_X = (NATIVE_IN != 0);
    constexpr bool NATIVE_R = (NATIVE_RESIDUAL != 0);
    constexpr bool BAND_X = (BAND != 0);
    constexpr bool RM = (IS_TILE == 0);
    static_assert(!BAND_X || IS_TILE == 0, "rms_norm_ttnn: the BAND scheme is ROW_MAJOR-only");
    static_assert(
        !BAND_X || PARTIAL_W == 0, "rms_norm_ttnn: the BAND scheme masks pad lanes by zero-staging, not by scaler");
    constexpr bool HAS_G = (HAS_GAMMA != 0);
    constexpr bool HAS_B = (HAS_BIAS != 0);
    constexpr bool HAS_R = (HAS_RESIDUAL != 0);
    constexpr bool PC_RM = (PER_CHANNEL_IS_RM != 0);
    constexpr bool PC_NARROW = (NARROW_PC_STAGE != 0);
    static_assert(!NATIVE_R || HAS_R, "rms_norm_ttnn: NATIVE_RESIDUAL without a residual");
    static_assert(!NATIVE_R || NATIVE_X, "rms_norm_ttnn: a zero-copy residual implies a zero-copy x");
    // A1: the residual carries the input's IDENTICAL shard spec by contract, so the
    // two native flags always agree.  They are SEPARATE flags anyway so that a
    // future divergence fails HERE rather than silently computing norm(x) instead
    // of norm(x + r) -- the reader's early return would otherwise skip a stream it
    // was supposed to fetch.
    static_assert(
        !NATIVE_X || !HAS_R || NATIVE_R,
        "rms_norm_ttnn: a zero-copy x with a non-resident residual would skip the residual's read");
    // X_RESIDENT == PER_CHANNEL_RESIDENT, from the descriptor's regime decision (D14).
    constexpr bool X_RESIDENT = (X_RES != 0);
    static_assert(NUM_W_CHUNKS > 1 || X_RESIDENT, "rms_norm_ttnn: a one-chunk width is resident by definition");
    // The whole point of the L5 regime: the activations are staged ONCE per
    // row-block, however many width chunks the derived CBs are cut into.
    constexpr uint32_t NUM_PASSES = X_RESIDENT ? 1 : 2;

    // Bytes of one full width chunk of a row-major stick, and of the last one
    // (short by the tile padding when W is not tile-aligned).
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(3);   // tile-rows owned by this core
    // Width slice this core owns (Lamp L1/L4).  On the whole-row schemes this is
    // (0, WT_CHUNK); under a width split it is the core's shard column range.
    const uint32_t w_start = get_arg_val<uint32_t>(4);
    const uint32_t w_real = get_arg_val<uint32_t>(5);  // REAL width tiles (<= WT_CHUNK)
    // The ROW_MAJOR view of the SAME slice: sticks and width ELEMENTS.  On the
    // tile-axis schemes these are derived host-side from (row_start, w_start) so
    // the two views cannot drift (_work_tile_axis); on the BAND scheme they are
    // the primary extents and are not tile-aligned on either axis.
    const uint32_t stick_base = get_arg_val<uint32_t>(6);    // first stick this core owns
    const uint32_t stick_count = get_arg_val<uint32_t>(7);   // sticks owned
    const uint32_t w_off_elems = get_arg_val<uint32_t>(8);   // first width element owned
    const uint32_t w_real_elems = get_arg_val<uint32_t>(9);  // REAL width elements owned
    // A2 / A1 -- appended, so indices 0..9 are the seed's exactly.
    const uint32_t bias_addr = get_arg_val<uint32_t>(10);
    const uint32_t residual_addr = get_arg_val<uint32_t>(11);

    // An INACTIVE core: it joined the program only so the width combine's stat
    // multicast lands in a cb_row_final this program owns (a width shard grid need
    // not be a rectangle, so the mcast box can be larger than the grid).  It holds
    // no shard, so it must not touch a shard-backed CB at all.  A4: the
    // zero-volume program is EVERY core taking this return, which is why it needs
    // no separate kernel -- and why the return sits above every accessor
    // construction and every CB touch.
    if (num_rows == 0) {
        return;
    }

    const auto x_acc = TensorAccessor(x_args, x_addr);

    // ---- boot, FIRST: hand the resident shard to the compute kernel ---------
    //
    // PERF 1.  This publish is a pure CB hand-off of data that is ALREADY in this
    // core's L1 (a zero-copy shard-backed ring): it reads nothing, moves nothing and
    // depends on nothing below it.  It used to sit AFTER the scaler boot and after the
    // per-channel operand's DRAM read, and `compute_square`'s `cb_wait_front` is blocked
    // on it -- so the compute kernel idled through both.  MEASURED on the pinned
    // `(1,1,32,7168)` WIDTH shard `[32,256]` `(7,4)` 28 cores (bf16 / HiFi2 /
    // fp32_dest_acc_en=False, UNCHANGED -- this move touches no precision knob and the
    // output is bit-identical, pcc 0.999985 both sides):
    //     reader_native_publish END   2358 ->  616 ns
    //     compute_reduce END          2740 -> 1289 ns
    //     writer_gather_ship END      3166 -> 1936 ns
    //     whole op                    5335 -> 4121 ns   = 1.295x
    // and 1.09x-1.35x across the other sharded geometries, 1.02x-1.04x interleaved,
    // flat (within noise) on STREAM / BAND / ragged-Wt, where there is no native shard
    // to publish.  The gamma DRAM latency now drains under pass A and the combine.
    //
    // ORDERING SAFETY.  `cb_scaler` and `cb_input_tiles` are different CBs, waited on
    // independently by the compute kernel, so nothing below reorders against this.  The
    // `num_rows == 0` early return still sits ABOVE every CB touch (A4).  A ragged
    // shard's pad-tile `async_write_zeros` inside `publish_native_shard` now runs before
    // any read is in flight, which is strictly safer than running it after.
    const uint32_t x_tile_bytes = get_tile_size(cb_input_tiles);

    if constexpr (NATIVE_X) {
        MaybeDeviceZoneScope("reader_native_publish");
        // A TEMPLATE on the CB id, not a lambda taking one: `cb_reserve_back` and
        // `DataflowBuffer` want a compile-time buffer index (a runtime one costs an
        // indexed lookup where the seed had an immediate), and this sits on the
        // measured native paths.  Two instantiations, one per stream.
        publish_native_shard<cb_input_tiles, WT_CHUNK, IN_SHARD_PAGES>(w_real, x_tile_bytes);
        if constexpr (NATIVE_R) {
            publish_native_shard<cb_residual_tiles, WT_CHUNK, IN_SHARD_PAGES>(w_real, x_tile_bytes);
        }
    }

    // ---- boot: what cb_scaler carries, per reduce datapath ----------------
    // Value is exactly 1.0 everywhere; 1/W is applied in fp32 by the compute
    // finalize, never folded into a bf16 scaler (R4).
    //
    //   ReduceTile       aligned : [full scaler]                   -> 1 tile
    //                    partial : [full scaler, partial scaler]   -> 2 tiles
    //   AccumulateViaAdd aligned : [scaler] (unused by the datapath, but keeps
    //                              the boot SrcB format real)      -> 1 tile
    //                    partial : [0/1 mask]                      -> 1 tile
    // The tile COUNT is the descriptor's SCALER_TILES, which the compute kernel
    // pops -- this branch must agree with it (asserted host-side).
    {
        MaybeDeviceZoneScope("reader_scaler_boot");
        if constexpr (REDUCE_ACC_VIA_ADD != 0) {
            if constexpr (PARTIAL_W != 0) {
                // 0/1 mask in the row-0 broadcast layout AccumulateViaAdd's masked
                // accumulating broadcast-mul consumes for the last width tile.
                dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ckernel::ReduceDim::REDUCE_ROW>(PARTIAL_W);
            } else {
                dataflow_kernel_lib::
                    prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
            }
        } else if constexpr (PARTIAL_W != 0) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                PARTIAL_W>(1.0f);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    }

    // ---- boot: establish the pad-lane invariant on the RM staging rings ----
    // A1: BOTH activation rings, for the reason in the file header -- an unzeroed
    // residual ring reintroduces exactly the NaN this zero exists to prevent, and
    // it does it inside cb_x_sum where only a FINITE pad lane can still be masked.
    if constexpr (RM && STAGE_ZERO != 0) {
        MaybeDeviceZoneScope("reader_stage_zero");
        Noc noc;
        DataflowBuffer stage_dfb(cb_input_sticks);
        noc.async_write_zeros(stage_dfb, stage_dfb.get_total_size_bytes());
        if constexpr (HAS_R) {
            DataflowBuffer res_dfb(cb_residual_sticks);
            noc.async_write_zeros(res_dfb, res_dfb.get_total_size_bytes());
        }
        noc.write_zeros_l1_barrier();
    }

    // ---- boot: the COMBINE's one-hot permutation bank (Perf 3, descriptor D27) ----
    //
    // WHAT IT IS.  Page r of cb_bank is E_r: a bf16 tile that is all zero except a
    // single EXACT 1.0 at element [0][r].  The compute kernel uses ONE bank for BOTH
    // directions of the combine's compact-partial permutation:
    //     pack    C = partial_r x E_r      E_r[0][r] = 1   -> C[i][r] = partial_r[i][0]
    //     unpack  C = compact   x E_r^T    (matmul's srcB `transpose` reads E_r as
    //                                       E_r^T, so E_r^T[r][0] = 1)
    //                                                     -> C[i][0] = compact[i][r]
    // so a core's BLOCK_ROWS column-shaped partials become BLOCK_ROWS COLUMNS of one
    // tile before the gather, and come back apart after the multicast.  That is what
    // makes the gather one whole-tile transaction, the root's fold ONE DEST window per
    // round instead of BLOCK_ROWS, and the landing ring flat in BLOCK_ROWS.
    //
    // WHY IT IS SYNTHESIZED HERE AND NOT PASSED AS A TENSOR.  It is a pure function of
    // BLOCK_ROWS, identical on every core, and this kernel already synthesizes the
    // reduce's constant tiles the same way (`reader_scaler_boot` above); a host tensor
    // would add an input-plumbing surface -- an extra tensor argument, its accessor, its
    // allocation and a DRAM read -- for a constant the device can write in L1.
    //
    // WHY bf16.  A one-hot is EXACT in bf16 (1.0 == 0x3F80), so the permutation matmuls
    // multiply by a true unit and the bank costs NOTHING in accuracy, at half the L1 of
    // an fp32 bank.  MEASURED perf-flat against fp32 (`member_pack` 419 vs 423 ns at
    // BLOCK_ROWS 8, 1033 vs 1038 at 32 -- inside noise), so bf16 is free.
    //
    // RAW-L1-STORE JUSTIFICATION.  The zeroing is the device zero API
    // (`Noc::async_write_zeros`), exactly as `reader_stage_zero` above; only the
    // BLOCK_ROWS single-element stores are hand-rolled.  No kernel_lib helper writes an
    // arbitrary constant at an arbitrary tile position: `l1_helpers.hpp` offers
    // `zero_tile` / `prepare_zero_tile` (used here for the zero half) and
    // `reduce_helpers_dataflow.hpp`'s `prepare_reduce_scaler` / `prepare_reduce_mask`
    // emit a UNIFORM scaler or a row-0 CONTIGUOUS 0/1 mask -- neither can place a single
    // 1.0 at column r.  Adding a helper for a one-off boot constant would be a worse
    // trade than 8 lines with the tile layout spelled out; the layout arithmetic is the
    // standard 2x2-faces-of-16x16 one and is written out below so it is checkable.
    if constexpr (BANK_PAGES != 0) {
        MaybeDeviceZoneScope("reader_bank_boot");
        Noc noc;
        DataflowBuffer bank_dfb(cb_bank);
        const uint32_t bank_tile_bytes = get_tile_size(cb_bank);
        // A fresh CB has write_ptr == base, so overload (1) of async_write_zeros (which
        // writes at the WRITE pointer) covers the whole bank in one call.  Zero FIRST,
        // barrier, THEN place the ones: the zero engine must not race the stores.
        bank_dfb.reserve_back(BANK_PAGES);
        noc.async_write_zeros(bank_dfb, BANK_PAGES * bank_tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t bank_base = get_write_ptr(cb_bank);
        for (uint32_t r = 0; r < BANK_PAGES; ++r) {
            // Element [0][r] of a 32x32 tile stored as 2x2 faces of 16x16: row 0 always
            // lands in the TOP face row, so the face is (r / 16) (face 0 for columns
            // 0..15, face 1 for 16..31) and the offset inside it is (r % 16) elements.
            const uint32_t face_bytes = bank_tile_bytes / 4;
            const uint32_t byte_off =
                r * bank_tile_bytes + (r / FACE_DIM) * face_bytes + (r % FACE_DIM) * sizeof(uint16_t);
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bank_base + byte_off) = BF16_ONE;
        }
        bank_dfb.push_back(BANK_PAGES);
    }

    // ---- the per-channel operands: one chunk's worth of tiles (or sticks) ----
    // In the RESIDENT regimes this runs once per core before the row-block loop
    // and the tiles are never popped; in STREAM it runs per pass-B chunk.
    // `w_start` shifts every index/offset by this core's width slice; on the
    // whole-row schemes it is 0.
    auto stage_per_channel = [&](uint32_t c) {
        const uint32_t first_wt = w_start + c * WT_CHUNK;
        if constexpr (HAS_G) {
            MaybeDeviceZoneScope("reader_read_gamma");
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            stage_per_channel_chunk<
                cb_gamma_sticks,
                cb_gamma_tiles,
                WT,
                WT_CHUNK,
                GAMMA_ELEM_BYTES,
                W_ELEMS,
                GAMMA_TRIM,
                PC_RM,
                (GAMMA_BLOCKED != 0),
                PC_NARROW>(g_acc, first_wt);
        }
        if constexpr (HAS_B) {
            MaybeDeviceZoneScope("reader_read_bias");
            const auto b_acc = TensorAccessor(bias_args, bias_addr);
            stage_per_channel_chunk<
                cb_bias_sticks,
                cb_bias_tiles,
                WT,
                WT_CHUNK,
                BIAS_ELEM_BYTES,
                W_ELEMS,
                BIAS_TRIM,
                PC_RM,
                (BIAS_BLOCKED != 0),
                PC_NARROW>(b_acc, first_wt);
        }
    };

    // Resident per-channel operands are staged ONCE per core, for every chunk the
    // row is cut into (NUM_W_CHUNKS == 1 in the RESIDENT regime, so this is one
    // call there).  In STREAM they are re-staged per pass-B chunk of every
    // row-block instead -- which for a prefill profile is as many DRAM bytes as x.
    if constexpr (X_RESIDENT) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            stage_per_channel(c);
        }
    }

    // ---- BAND staging: a core's own resident RM shard -> the tilize ring -----
    // The band is `band_bytes` of every stick it owns, at base + local_stick *
    // SHARD_ROW_BYTES; the ring wants it at the padded tile-column stride.  When
    // the band fills its tile columns exactly AND the shard stride matches, a
    // whole tile-row moves in ONE transaction; otherwise it is one per stick --
    // the same granularity the accessor path uses, but out of local L1 rather
    // than DRAM.  Trailing lanes keep the boot zero, so they add exactly 0 to
    // sum(t^2) and no reduce mask is needed (STAGE_ZERO / PARTIAL_W == 0).
    //
    // A1: `base` is a parameter, so the residual's band -- identical shard spec,
    // identical stride -- stages through this same code from ITS own L1.
    const uint32_t band_bytes = w_real_elems * ELEM_BYTES;
    // The band is staged in the tensor's GLOBAL TILE FRAME: its first element sits
    // at lane (w_off_elems % 32) of the staged stick, so staged tile column j IS
    // global width tile (w_off_elems / 32 + j).  Two things fall out of that, and
    // both are why the frame is not the band's own byte offset:
    //   * a per-channel operand (RM or TILE) is fetched at a tile-column offset,
    //     which is a multiple of 64 bytes for every dtype -- an unaligned DRAM
    //     read is silently truncated to the alignment, so an offset of "the band's
    //     first element" would hand three cores in four the WRONG weights;
    //   * the leading lanes [0, delta) and the trailing ones are the boot zeros, so
    //     they contribute exactly 0 to sum(t^2) -- no reduce mask, either side.
    // The shard granule keeps both the L1 source and the shifted destination
    // 16-byte aligned (w_off_elems is a multiple of L1_align/elem_size).
    const uint32_t band_delta_bytes = (w_off_elems % TILE_DIM) * ELEM_BYTES;
    const bool band_contiguous =
        (band_delta_bytes == 0) && (band_bytes == CHUNK_ROW_BYTES) && (SHARD_ROW_BYTES == CHUNK_ROW_BYTES);
    auto stage_band = [&](uint32_t cb, uint32_t base, uint32_t stick_start, uint32_t sticks) {
        for (uint32_t s = 0; s < sticks; s += TILE_DIM) {
            const uint32_t n = ((sticks - s) < TILE_DIM) ? (sticks - s) : TILE_DIM;
            cb_reserve_back(cb, WT_CHUNK);
            const uint32_t dst = get_write_ptr(cb) + band_delta_bytes;
            const uint32_t src = base + (stick_start + s - stick_base) * SHARD_ROW_BYTES;
            if (band_bytes != 0) {
                if (band_contiguous) {
                    noc_async_read(get_noc_addr(src), dst, n * CHUNK_ROW_BYTES);
                } else {
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_read(get_noc_addr(src + i * SHARD_ROW_BYTES), dst + i * CHUNK_ROW_BYTES, band_bytes);
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb, WT_CHUNK);
        }
    };

    // ---- one width chunk of one row-block, BOTH activation streams ----------
    // Transaction granularity is WT_CHUNK tiles (one tile-row of the chunk): a
    // single knob-derived unit that divides every CB ring by construction, and
    // >= 4 tiles per barrier whenever the block allows it.
    //
    // A1: on the TILE path the two streams are issued inside the SAME width-tile
    // loop and drained by the SAME barrier, because noc_async_read_barrier fences
    // NoC-0 globally -- so a residual costs zero extra barriers and the two rings
    // stay in lockstep, which is what keeps the consumer's two Upfront waits from
    // deadlocking against each other.
    // R3 lever 3: the group is TXN_ROWS tile-rows -- ONE reserve, one issue run, ONE
    // barrier and one push per group instead of per tile-row.  The final group of a
    // ragged block is short and reserves/pushes only what it carries, which is the
    // ragged-tail rule: the ACTUAL page count, never the nominal one.
    auto stage_tile_rows = [&](auto&& xa, auto&& ra, uint32_t first_tile_row, uint32_t rows, uint32_t c) {
        for (uint32_t r = 0; r < rows; r += TXN_ROWS) {
            const uint32_t n = ((rows - r) < TXN_ROWS) ? (rows - r) : TXN_ROWS;
            const uint32_t pages = n * WT_CHUNK;
            uint32_t xl1 = 0;
            uint32_t rl1 = 0;
            if constexpr (!NATIVE_X) {
                cb_reserve_back(cb_input_tiles, pages);
                xl1 = get_write_ptr(cb_input_tiles);
            }
            if constexpr (HAS_R && !NATIVE_R) {
                cb_reserve_back(cb_residual_tiles, pages);
                rl1 = get_write_ptr(cb_residual_tiles);
            }
            // D33: the ragged tail chunk's trailing WT_PAD tiles are OUTSIDE the row.
            // Zero them (device zero API, on the pages just reserved) BEFORE issuing
            // any read: the zero borrows the write command buffer and is released only
            // by its own barrier, so it must not be interleaved with the real traffic.
            // `w_lim` then stops the read loop at the real tiles.  Both fold away at
            // WT_PAD == 0, which is every divisor build.
            const uint32_t w_lim = (HAS_WPAD && c + 1 == NUM_W_CHUNKS) ? WT_REAL_TAIL : WT_CHUNK;
            if constexpr (HAS_WPAD) {
                if (w_lim != WT_CHUNK) {
                    Noc noc;
                    const uint32_t pad_bytes = (WT_CHUNK - w_lim) * x_tile_bytes;
                    for (uint32_t g = 0; g < n; ++g) {
                        const uint32_t off = (g * WT_CHUNK + w_lim) * x_tile_bytes;
                        if constexpr (!NATIVE_X) {
                            DataflowBuffer xdfb(cb_input_tiles);
                            noc.async_write_zeros(xdfb, pad_bytes, {.offset_bytes = off});
                        }
                        if constexpr (HAS_R && !NATIVE_R) {
                            DataflowBuffer rdfb(cb_residual_tiles);
                            noc.async_write_zeros(rdfb, pad_bytes, {.offset_bytes = off});
                        }
                    }
                    noc.write_zeros_l1_barrier();
                }
            }
            for (uint32_t g = 0; g < n; ++g) {
                // + w_start: this core's width slice under a cross-core width split
                // (0 on the whole-row schemes).
                const uint32_t tile_base = (first_tile_row + r + g) * WT + w_start + c * WT_CHUNK;
                for (uint32_t w = 0; w < w_lim; ++w) {
                    if constexpr (!NATIVE_X) {
#ifndef RMS_ABLATE_READ_X
                        noc_async_read_tile(tile_base + w, xa, xl1);
#endif
                        xl1 += x_tile_bytes;
                    }
                    if constexpr (HAS_R && !NATIVE_R) {
#ifndef RMS_ABLATE_READ_X
                        noc_async_read_tile(tile_base + w, ra, rl1);
#endif
                        rl1 += x_tile_bytes;
                    }
                }
                // Step over the (already zeroed) pad pages so the next tile-row starts
                // at its own WT_CHUNK-aligned base.  Zero-width off the ragged path.
                xl1 += (WT_CHUNK - w_lim) * x_tile_bytes;
                rl1 += (WT_CHUNK - w_lim) * x_tile_bytes;
            }
            noc_async_read_barrier();
            if constexpr (!NATIVE_X) {
                cb_push_back(cb_input_tiles, pages);
            }
            if constexpr (HAS_R && !NATIVE_R) {
                cb_push_back(cb_residual_tiles, pages);
            }
        }
    };

    auto stage_activations_chunk = [&](uint32_t r0, uint32_t rows, uint32_t c) {
        MaybeDeviceZoneScope("reader_read_x");
        const uint32_t first_tile_row = row_start + r0;
        if constexpr (NATIVE_X) {
            // Both streams (the static_assert above pins them together) are already
            // resident and were published above -- no NoC read at all.
            return;
        } else if constexpr (RM) {
            const uint32_t stick_start = stick_base + r0 * TILE_DIM;
            uint32_t sticks = rows * TILE_DIM;
            if (r0 * TILE_DIM + sticks > stick_count) {
                sticks = stick_count - r0 * TILE_DIM;  // short final tile-row
            }
            if constexpr (BAND_X) {
                stage_band(cb_input_sticks, x_addr, stick_start, sticks);
                if constexpr (HAS_R) {
                    stage_band(cb_residual_sticks, residual_addr, stick_start, sticks);
                }
            } else {
                // D33: the ragged tail stages raw at the PADDED stride (see
                // `stage_ragged_tail_sticks`); every other chunk is the helper's.
                if constexpr (HAS_WPAD) {
                    if (c + 1 == NUM_W_CHUNKS) {
                        stage_ragged_tail_sticks<cb_input_sticks, WT_CHUNK, CHUNK_ROW_BYTES, LAST_CHUNK_ROW_BYTES>(
                            x_acc, sticks, stick_start, c * CHUNK_ROW_BYTES);
                        if constexpr (HAS_R) {
                            const auto r_acc = TensorAccessor(residual_args, residual_addr);
                            stage_ragged_tail_sticks<
                                cb_residual_sticks,
                                WT_CHUNK,
                                CHUNK_ROW_BYTES,
                                LAST_CHUNK_ROW_BYTES>(r_acc, sticks, stick_start, c * CHUNK_ROW_BYTES);
                        }
                        return;
                    }
                }
                const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? LAST_CHUNK_ROW_BYTES : CHUNK_ROW_BYTES;
                dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                    x_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
                if constexpr (HAS_R) {
                    // Its own barrier: the helper owns its reserve/push, so the two
                    // RM streams cannot share one the way the TILE path's do.
                    const auto r_acc = TensorAccessor(residual_args, residual_addr);
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_residual_sticks>(
                        r_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
                }
            }
        } else if constexpr (HAS_R && !NATIVE_R) {
            const auto r_acc = TensorAccessor(residual_args, residual_addr);
            stage_tile_rows(x_acc, r_acc, first_tile_row, rows, c);
        } else {
            stage_tile_rows(x_acc, x_acc, first_tile_row, rows, c);
        }
    };

    // ---- row-block loop ---------------------------------------------------
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
            for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
                stage_activations_chunk(r0, rows, c);
                // STREAM: the per-channel operands are chunked and re-read for
                // every pass-B chunk.
                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_per_channel(c);
                    }
                }
            }
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (regimes R1 and R3, op_design.md -> Dataflow Strategy, store_chunk + γ supply).
//
// Every core owns a W-slice [w0, w0 + V) of its tile-rows (R1: the whole row). γ tile t of the slice is γ tile
// w0 + t of the tensor; output tile t of the slice is page tile_row * Wt + w0 + t. Sharded (R3): cb_output IS
// the output shard (compute packs into it) — this RISC-V only supplies γ.
//
// γ supply (has_gamma) — on this RISC-V, not the reader's (perf lamp L4): the writer is idle until the
// first output chunk, and keeping γ off the reader lets it prefetch block k+1 while block k is scaled.
//   fill (once per staging-row chunk, on first use): cb_gamma_row <- γ values in a compact staging row
//   (never pushed):
//     RM γ   : one NoC read of the W-element stick (page = 32*es_g B per γ tile).
//     TILE γ : per γ tile, its row-0 segments (gamma_segment_offset: face 0 row 0, face 1 row 0 — and for
//              Bfp8_b also their shared exponents), each into its own GAMMA_SEG_SLOT-byte slot (page =
//              gamma_num_segs slots per γ tile). Rows >= 1 are never read by BroadcastDim::Row, so only
//              row 0 moves (1/16 of the tile's bytes at bf16).
//   expand: per γ tile, the segments are copied by local L1->L1 NoC reads to the same byte offsets of a
//     cb_gamma_tiles page — row 0 of faces 0 / 1, the only rows the BroadcastDim::Row multiply consumes.
//     gamma_resident: all Wt tiles once, in one Wp reservation pushed C tiles at a time (fill of chunk
//                     c+1 overlaps expansion of chunk c); compute waits cumulatively and keeps the
//                     window for the whole kernel.
//     gamma_direct  : (TILE γ, resident) no staging row and no expansion — each γ tile's row-0 byte
//                     prefix [0, gamma_prefix_bytes) (every row-0 segment, in place) is read from DRAM
//                     straight into its cb_gamma_tiles page: ONE NoC read per γ tile instead of
//                     2·gamma_num_segs. Source (tile base) and destination (page base) are both
//                     tile-aligned, so their low address bits match. Rows >= 1 of face 0 come along and
//                     are never read by BroadcastDim::Row.
//     streaming direct: (TILE γ, L1 fallback when the staging row does not fit — e.g. a huge resident shard)
//                     the direct read per (block, chunk) step, no staging row.
//     streaming     : per (block, chunk) step, C tiles; step s+1 is pushed BEFORE output step s is
//                     drained, so compute always finds the next chunk ready. Deadlock-free: the
//                     reserve for step s+1 needs step s-1 popped, which compute did (end of
//                     scale_by_gamma(s-1)) before it produced output step s-1, already drained here.
//
// Output: consumes cb_output in nominal C-page quanta, in the order compute produces them:
//   per block, per chunk c, per tile-row r of the block: C pages.
//   TILE: write the chunk's valid tiles (page id tile_row*Wt + w), skip pad tiles.
//   RM:   full chunks via write_sticks_after_untilize; the tail chunk is custom (C-wide stride in L1,
//         only the valid tail bytes of each valid stick are written).
// One write flush per C-page quantum (slot reuse); a single ack barrier at kernel end.
//
// Raw NoC / CB calls for γ: no kernel_lib helper extracts row 0 of a tile or builds a row-0-only tile.
//
// γ multicast (Perf 1, perf_experiments/gamma_mcast): γ read ONCE per group of Tensix cores that share
// the same γ window, multicast into every member's landing region at the same L1 address (kernel_lib McastArgs
// channel "gmcast", hardware multicast on this RISC-V's NoC, Counter data-ready, no receiver handshake: the
// landing region is written exactly once per program, at distinct addresses per round, so no slot is reused).
//   resident direct (TILE γ): landing = cb_gamma_tiles; round c = chunk c (the injector reads chunk c+1 from DRAM
//                             while chunk c is multicast); every member pushes chunk c once it has landed.
//   staging row (RM γ / TILE γ expand path): landing = cb_gamma_row; ONE round (the injector fills the whole
//                             staging row, multicasts it, every member expands locally as before).
// Absent channel (no γ, a group of one, streaming-direct L1 fallback) = the baseline per-core reads.

#include <stdint.h>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_combine.hpp"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "rms_norm_mcast_roles.hpp"

// Measurement-only payload ablation (host RMS_NORM_ABLATE; 0 in production): NoC data movement skipped, every
// CB handshake and barrier kept.
#ifndef RMS_ABLATE_WRITES
#define RMS_ABLATE_WRITES 0
#endif
#ifndef RMS_ABLATE_GAMMA
#define RMS_ABLATE_GAMMA 0
#endif

// Stage zones (permanent; MaybeDeviceZoneScope, kernel_lib/perf_instrumentation.hpp):
//   wr_gamma        the γ supply prologue (resident window / first streaming chunk), issue + barriers
//   wr_wait_out     per (block, chunk, tile-row): cb_output wait (starved on compute)
//   wr_issue        per (block, chunk, tile-row): the output write issue loop + flush
//   wr_final_barrier the kernel-end write ack barrier

namespace {
constexpr uint32_t cb_gamma_row = 5;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_output = 16;
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t TILE_HW = TILE_DIM * TILE_DIM;
constexpr uint32_t BFP8_EXP_BYTES = 64;  // Bfp8_b: exponent section (4 faces x 16 face rows, 1 B each)
// gamma_direct CT arg values — mirror GAMMA_DIRECT_* on the host.
constexpr uint32_t GAMMA_DIRECT_OFF = 0;
constexpr uint32_t GAMMA_DIRECT_PREFIX = 1;
constexpr uint32_t GAMMA_DIRECT_SEGMENTS = 2;

// Row-0 segment geometry of a γ tile — mirrors gamma_row_segments() / gamma_segment_bytes() on the host.
// Only row 0 of faces 0 and 1 is consumed by the BroadcastDim::Row multiply.
//   Float32 / Float16_b: 2 segments of 16 datums, at tile offsets 0 and one face.
//   Bfp8_b (format-aware): exponents [0, 32) (face 0/1 row exponents at bytes 0 / 16), mantissa
//     row 0 of face 0 at 64 and of face 1 at 64 + 256; 32 B each (the second 16 B are never read).
constexpr uint32_t T_g = get_tile_size(cb_gamma_tiles);
constexpr bool gamma_is_bfp8 = T_g == BFP8_EXP_BYTES + TILE_HW;
// (T_g is meaningless when there is no γ: cb_gamma_tiles is then not configured.)
constexpr uint32_t es_g = gamma_is_bfp8 ? 1 : T_g / TILE_HW;
constexpr uint32_t face_bytes_g = FACE_DIM * FACE_DIM * es_g;
constexpr uint32_t gamma_num_segs = gamma_is_bfp8 ? 3 : 2;
constexpr uint32_t gamma_seg_bytes = gamma_is_bfp8 ? 2 * FACE_DIM : FACE_DIM * es_g;
constexpr uint32_t gamma_segment_offset(uint32_t k) {
    if constexpr (gamma_is_bfp8) {
        return k == 0 ? 0 : BFP8_EXP_BYTES + (k - 1) * face_bytes_g;
    }
    return k * face_bytes_g;
}
// Bytes [0, n) of a γ tile holding every row-0 segment — mirrors gamma_row0_prefix_bytes() on the host.
constexpr uint32_t gamma_prefix_bytes = gamma_segment_offset(gamma_num_segs - 1) + gamma_seg_bytes;

// Bank stagger (perf lamp L9) — compile-time switch RMS_BANK_STAGGER (host define, 0/1). On: a walk over
// [0, n) starts at this core's bank offset s (< NUM_DRAM_BANKS) and wraps — same transactions, same L1
// slots, only the issue order rotates, so cores in lockstep do not all hit one DRAM bank (every tile-row
// starts on the same bank when Wt % NUM_DRAM_BANKS == 0). Off: the plain in-order loop, no extra work.
constexpr bool bank_stagger_on = RMS_BANK_STAGGER != 0;
FORCE_INLINE uint32_t stagger_first(uint32_t s, uint32_t n) {
    if constexpr (bank_stagger_on) {
        return s < n ? s : s % n;
    }
    return 0;
}
FORCE_INLINE uint32_t stagger_next(uint32_t j, uint32_t n) {
    if constexpr (bank_stagger_on) {
        return (j + 1 == n) ? 0 : j + 1;
    }
    return j + 1;
}

// Copy the row-0 segments of γ tiles [first, first + n) from the staging row into consecutive
// cb_gamma_tiles pages at l1_dst (local L1->L1 NoC reads, all 16 B-aligned, stateful single-packet
// issue). No barrier.
template <uint32_t gamma_row_page>
FORCE_INLINE void expand_gamma_tiles(uint32_t gamma_row_addr, uint32_t l1_dst, uint32_t first, uint32_t n) {
    // Staging slot k sits k slots into the page: RM 16 elements, TILE one GAMMA_SEG_SLOT.
    constexpr uint32_t slot = gamma_row_page / gamma_num_segs;
    if (gamma_row_addr % 16 != 0) {
        // A ROW_MAJOR-sharded slice whose first γ element is not 16 B-aligned in the staging row (its slice
        // width is a multiple of the shard granule, not of 16 B of γ): the NoC cannot copy between
        // mismatched low address bits, so the 2 x 16 row-0 datums per tile are copied by 16-bit loads.
        constexpr uint32_t halfwords = gamma_seg_bytes / 2;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t src = gamma_row_addr + (first + j) * gamma_row_page;
            for (uint32_t k = 0; k < gamma_num_segs; ++k) {
                volatile tt_l1_ptr uint16_t* d =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_dst + j * T_g + gamma_segment_offset(k));
                volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src + k * slot);
                for (uint32_t h = 0; h < halfwords; ++h) {
                    d[h] = sp[h];
                }
            }
        }
        return;
    }
    // Same source core and size for every read: program the read command buffer once and issue each
    // segment with two address writes (per-read issue cost is what binds when this runs on the writer
    // next to the output stream). Re-set per call: the TILE γ fill's DRAM reads reprogram it.
    noc_async_read_one_packet_set_state(get_noc_addr(gamma_row_addr), gamma_seg_bytes);
    uint32_t src = gamma_row_addr + first * gamma_row_page;
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t k = 0; k < gamma_num_segs; ++k) {
            noc_async_read_one_packet_with_state(src + k * slot, l1_dst + gamma_segment_offset(k));
        }
        src += gamma_row_page;
        l1_dst += T_g;
    }
}
// Writer-side all-gather (Perf 1, perf_experiments/combine_pipeline; AG_WRITER): the wave all-gather of the
// reduce-scatter combine on THIS RISC-V (its own NoC), so the reader's partial sends and the finalized-row all-gather
// proceed concurrently on the two NoCs instead of sharing one RISC-V's issue loop and write barriers. Same writes as
// the reader's step C (rms_norm_reader.cpp pipe_block): owner i of wave k sends its q_w finalized rows (cb_rms_root,
// the only consumer is now this RISC-V) to row k*Rw + i*q_w of every member's cb_inv_rms, write barrier, then pushes a
// token to the reader, which increments inv sem k of every other member (see below for why the atomics are not issued
// from this RISC-V).
template <uint32_t G, uint32_t T_interm, uint32_t BLOCK_ROWS, uint32_t RW_FULL, uint32_t QW_FULL, uint32_t WAVES_FULL>
FORCE_INLINE void allgather_block(
    uint32_t inv_sem_base,
    uint32_t inv_base,
    uint32_t rms_root_base,
    uint32_t member_arg_base,
    uint32_t my_index,
    uint32_t rows_in_block) {
    constexpr uint32_t cb_rms_root = 11;
    constexpr uint32_t cb_ag_token = 15;  // -> reader: "wave all-gathered" tokens
    // (instantiated with zeros when AG_WRITER is off: kernel_main is not a template, so guard the divisions)
    constexpr uint32_t H = QW_FULL > 0 ? (RW_FULL + QW_FULL - 1) / QW_FULL : 0;
    constexpr uint32_t rms_root_pages = WAVES_FULL * QW_FULL;
    if constexpr (QW_FULL == 0 || WAVES_FULL == 0) {
        return;
    } else {
        if (my_index >= H) {
            return;
        }
        auto member_x = [member_arg_base](uint32_t k) { return get_arg_val<uint32_t>(member_arg_base + k) >> 16; };
        auto member_y = [member_arg_base](uint32_t k) { return get_arg_val<uint32_t>(member_arg_base + k) & 0xFFFF; };
        uint32_t QW = QW_FULL;
        if (rows_in_block != BLOCK_ROWS) {
            const uint32_t q = ((rows_in_block + H - 1) / H + WAVES_FULL - 1) / WAVES_FULL;
            QW = q < QW_FULL ? q : QW_FULL;
        }
        const uint32_t RW = QW * H;
        const uint32_t waves = (rows_in_block + RW - 1) / RW;
        for (uint32_t k = 0; k < waves; ++k) {
            const uint32_t n = (rows_in_block - k * RW) < RW ? rows_in_block - k * RW : RW;
            const uint32_t first = my_index * QW;
            if (first >= n) {
                continue;
            }
            const uint32_t mine = (n - first) < QW ? n - first : QW;
            {
                MaybeDeviceZoneScope("wr_wait_root");
                cb_wait_front(cb_rms_root, mine);
            }
            MaybeDeviceZoneScope("wr_allgather");
            const uint32_t rows_src = get_read_ptr(cb_rms_root);
            const uint32_t to_end = (rms_root_base + rms_root_pages * T_interm - rows_src) / T_interm;
            const uint32_t run0 = mine < to_end ? mine : to_end;
            const uint32_t dst = inv_base + (k * RW + first) * T_interm;
            for (uint32_t mm = 0; mm < G; ++mm) {
                const uint32_t m = rms_norm_combine::rotated(mm, my_index, G);
                rms_norm_combine::combine_write_tiles<T_interm>(rows_src, member_x(m), member_y(m), dst, run0);
                if (run0 < mine) {
                    rms_norm_combine::combine_write_tiles<T_interm>(
                        rms_root_base, member_x(m), member_y(m), dst + run0 * T_interm, mine - run0);
                }
            }
            noc_async_write_barrier();
            // The arrival increments are NOT issued here: a NoC atomic from this RISC-V (BRISC) trips the debug
            // assert noc_assert_targ_addr_mid_clear on its atomic command buffer (firmware leaves a PCIe routing value
            // in NOC_TARG_ADDR_MID there), and clearing it would misroute that firmware traffic. The rows have landed
            // (write barrier above); a local token tells the reader (NCRISC), which issues the increments on its own
            // NoC. Tokens are pushed in this core's owned-wave order — the order the reader consumes them.
            cb_reserve_back(cb_ag_token, 1);
            cb_push_back(cb_ag_token, 1);
            for (uint32_t r = 0; r < mine; ++r) {
                cb_pop_front(cb_rms_root, 1);
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr bool is_rm = get_compile_time_arg_val(0) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t W = get_compile_time_arg_val(6);
    constexpr uint32_t num_sticks = get_compile_time_arg_val(7);
    constexpr bool gamma_is_tile = get_compile_time_arg_val(10) != 0;
    constexpr bool gamma_resident = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t gamma_row_page = get_compile_time_arg_val(12);
    // TILE γ resident: read row 0 straight into cb_gamma_tiles (no staging row, no expansion) —
    // GAMMA_DIRECT_PREFIX: one read of the row-0 byte prefix; GAMMA_DIRECT_SEGMENTS: one read per segment.
    constexpr uint32_t gamma_direct = get_compile_time_arg_val(13);
    // Streaming γ takes the direct read only when the host's L1 fallback chose it (the staging row does not fit).
    static_assert(gamma_direct == GAMMA_DIRECT_OFF || gamma_is_tile, "direct γ read: TILE γ only");
    constexpr bool sharded = get_compile_time_arg_val(14) != 0;
    constexpr bool rm_sharded = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t shard_row_bytes = get_compile_time_arg_val(21);  // ROW_MAJOR shard stick bytes
    constexpr auto output_args = TensorAccessorArgs<22>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    // Writer-side wave all-gather (Perf 1, perf_experiments/combine_pipeline) (0 = off) + its geometry, after the γ
    // accessor args.
    constexpr uint32_t ag_ct = gamma_args.next_compile_time_args_offset();
    constexpr bool ag_writer = get_compile_time_arg_val(ag_ct) != 0;
    constexpr uint32_t ag_G = get_compile_time_arg_val(ag_ct + 1);
    constexpr uint32_t ag_wave_rows = get_compile_time_arg_val(ag_ct + 2);
    constexpr uint32_t ag_wave_q = get_compile_time_arg_val(ag_ct + 3);
    constexpr uint32_t ag_waves_full = get_compile_time_arg_val(ag_ct + 4);
    constexpr uint32_t ag_inv_sem_base = get_compile_time_arg_val(ag_ct + 5);
    constexpr uint32_t ag_member_arg_base = get_compile_time_arg_val(ag_ct + 6);
    constexpr uint32_t ag_T_interm = get_compile_time_arg_val(ag_ct + 7);
    constexpr auto gmcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("gmcast_ct_offset"),
        get_named_compile_time_arg_val("gmcast_rt_offset")>();
    // γ multicast channel present: this core's γ window is shared with its group (host gamma_mcast_groups()).
    constexpr bool gamma_mcast = decltype(gmcast)::active;
    [[maybe_unused]] Noc noc;

    constexpr uint32_t C = w_chunk_tiles;
    constexpr uint32_t Wp = num_w_chunks * C;
    constexpr uint32_t T_out = get_tile_size(cb_output);
    constexpr uint32_t es = T_out / (TILE_DIM * TILE_DIM);
    constexpr uint32_t chunk_stick_bytes = C * TILE_DIM * es;
    constexpr uint32_t tail_stick_bytes = (W - (num_w_chunks - 1) * C * TILE_DIM) * es;
    constexpr bool tail_chunk_is_full = tail_stick_bytes == chunk_stick_bytes;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t rows_this_core = get_arg_val<uint32_t>(3);
    // This core's DRAM bank offset (only read when RMS_BANK_STAGGER is on): output writes and γ reads rotate
    // inside each C-page quantum.
    const uint32_t bank_stagger = bank_stagger_on ? get_arg_val<uint32_t>(4) % NUM_DRAM_BANKS : 0;
    const uint32_t slice_w0 = get_arg_val<uint32_t>(5);     // first tile of this core's W-slice
    const uint32_t slice_valid = get_arg_val<uint32_t>(6);  // V: valid tiles of the slice
    [[maybe_unused]] const uint32_t slice_e0 = get_arg_val<uint32_t>(7);     // first element of the slice
    [[maybe_unused]] const uint32_t slice_elems = get_arg_val<uint32_t>(8);  // valid elements of the slice
    [[maybe_unused]] const uint32_t stick_count = get_arg_val<uint32_t>(9);  // ROW_MAJOR shard: valid sticks
    // Valid slice tiles of C-tile chunk c (0 for a chunk made only of pad tiles).
    auto chunk_valid = [slice_valid](uint32_t c) -> uint32_t {
        const uint32_t first = c * C;
        const uint32_t left = slice_valid > first ? slice_valid - first : 0;
        return left < C ? left : C;
    };

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const uint32_t num_blocks = (rows_this_core + block_rows - 1) / block_rows;

    // ---- γ supply: staging row fill + the first expansion(s) ----
    [[maybe_unused]] uint32_t gamma_row_addr = 0;
    [[maybe_unused]] uint32_t gamma_chunks_filled = 0;  // staging-row chunks whose bytes are issued
    [[maybe_unused]] const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
    // Issue (no barrier) the staging-row fill of chunk c. RM γ is filled whole up front instead.
    [[maybe_unused]] auto fill_gamma_chunk = [&](uint32_t c) {
        if constexpr (gamma_is_tile) {
            // Row-0 segments of each γ tile. Source offsets (gamma_segment_offset) and slot offsets are
            // multiples of the DRAM read alignment, so source and destination low bits match.
            constexpr uint32_t slot = gamma_row_page / gamma_num_segs;
            const uint32_t valid = chunk_valid(c);
            for (uint32_t t = c * C; t < c * C + valid; ++t) {
                const uint32_t dst = gamma_row_addr + t * gamma_row_page;
                for (uint32_t k = 0; k < gamma_num_segs; ++k) {
                    noc_async_read(
                        gamma_accessor.get_noc_addr(slice_w0 + t, gamma_segment_offset(k)),
                        dst + k * slot,
                        gamma_seg_bytes);
                }
            }
        }
        gamma_chunks_filled = c + 1;
    };
    // Direct read (no barrier) of the valid γ tiles of chunk c into the cb_gamma_tiles pages at l1_chunk: each
    // tile's row-0 byte prefix (GAMMA_DIRECT_PREFIX) or its row-0 segments (GAMMA_DIRECT_SEGMENTS), at the
    // page's matching byte offsets. Pad tiles (tail chunk only) are left as-is: they only multiply x pad tiles.
    [[maybe_unused]] auto read_gamma_chunk_direct = [&](uint32_t l1_chunk, uint32_t c) {
        const uint32_t valid = chunk_valid(c);
        uint32_t j = stagger_first(bank_stagger, valid);
        for (uint32_t n = 0; n < valid; ++n) {
            const uint32_t t = c * C + j;
            const uint32_t l1_tile = l1_chunk + j * T_g;
            if constexpr (RMS_ABLATE_GAMMA) {
            } else if constexpr (gamma_direct == GAMMA_DIRECT_PREFIX) {
                noc_async_read(gamma_accessor.get_noc_addr(slice_w0 + t), l1_tile, gamma_prefix_bytes);
            } else {
                for (uint32_t k = 0; k < gamma_num_segs; ++k) {
                    noc_async_read(
                        gamma_accessor.get_noc_addr(slice_w0 + t, gamma_segment_offset(k)),
                        l1_tile + gamma_segment_offset(k),
                        gamma_seg_bytes);
                }
            }
            j = stagger_next(j, valid);
        }
    };
    // Streaming γ: expand and push the C-tile chunk c (pad tiles beyond Wt are left as-is: they only
    // multiply x pad tiles, whose output tiles are never written). TILE γ staging is filled lazily, one
    // chunk ahead (block 0 requests chunks in order), so no fill is a long barrier on the writer.
    [[maybe_unused]] auto push_gamma_chunk = [&](uint32_t c) {
        const uint32_t valid = chunk_valid(c);
        cb_reserve_back(cb_gamma_tiles, C);
        if constexpr (gamma_direct != GAMMA_DIRECT_OFF) {
            // L1 fallback (no staging row): re-read row 0 of the chunk's γ tiles from DRAM every step.
            read_gamma_chunk_direct(get_write_ptr(cb_gamma_tiles), c);
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiles, C);
            return;
        }
        if (c >= gamma_chunks_filled) {
            fill_gamma_chunk(c);
        }
        noc_async_read_barrier();  // chunk c's staging bytes have landed (issued here or one step earlier)
        expand_gamma_tiles<gamma_row_page>(gamma_row_addr, get_write_ptr(cb_gamma_tiles), c * C, valid);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, C);
        // Lazy fill, one step ahead: chunk c+1's staging bytes land while output step c drains.
        if (c + 1 < num_w_chunks && c + 1 >= gamma_chunks_filled) {
            fill_gamma_chunk(c + 1);
        }
    };
    {
        MaybeDeviceZoneScope("wr_gamma");
        if constexpr (has_gamma && gamma_direct != GAMMA_DIRECT_OFF && gamma_resident) {
            if (num_blocks > 0) {
                // One reservation of the whole window, pushed chunk by chunk (compute waits cumulatively).
                cb_reserve_back(cb_gamma_tiles, Wp);
                const uint32_t gamma_tiles_base = get_write_ptr(cb_gamma_tiles);
                if constexpr (gamma_mcast) {
                    // Bytes of chunk c that matter: every valid tile's page up to the last one's row-0 prefix.
                    auto chunk_bytes = [&](uint32_t c) -> uint32_t {
                        const uint32_t v = chunk_valid(c);
                        return v == 0 ? 0 : (v - 1) * T_g + gamma_prefix_bytes;
                    };
                    [&](const auto& mc) {
                        if constexpr (std::decay_t<decltype(mc)>::active) {
                            if (mc.can_send()) {
                                auto sender = rms_norm_mcast::sender(mc, noc);
                                read_gamma_chunk_direct(gamma_tiles_base, 0);
                                noc_async_read_barrier();
                                for (uint32_t c = 0; c < num_w_chunks; ++c) {
                                    const uint32_t l1_c = gamma_tiles_base + c * C * T_g;
                                    if (c + 1 < num_w_chunks) {
                                        read_gamma_chunk_direct(
                                            l1_c + C * T_g, c + 1);  // in flight during the multicast
                                    }
                                    cb_push_back(cb_gamma_tiles, C);  // own copy landed (read barrier above)
                                    const uint32_t bytes = chunk_bytes(c);
                                    if (bytes > 0) {
                                        sender.send(l1_c, l1_c, bytes);
                                    } else {
                                        sender.send_signal();
                                    }
                                    noc_async_read_barrier();
                                }
                            } else {
                                auto receiver = rms_norm_mcast::receiver(mc, noc);
                                for (uint32_t c = 0; c < num_w_chunks; ++c) {
                                    receiver.receive(c);
                                    cb_push_back(cb_gamma_tiles, C);
                                }
                            }
                        }
                    }(gmcast);
                } else {
                    for (uint32_t c = 0; c < num_w_chunks; ++c) {
                        read_gamma_chunk_direct(gamma_tiles_base + c * C * T_g, c);
                        noc_async_read_barrier();
                        cb_push_back(cb_gamma_tiles, C);
                    }
                }
            }
        } else if constexpr (has_gamma) {
            if constexpr (gamma_direct == GAMMA_DIRECT_OFF) {
                gamma_row_addr = get_write_ptr(cb_gamma_row);
            }
            if constexpr (!gamma_is_tile) {
                static_assert(!has_gamma || gamma_is_tile || !gamma_is_bfp8, "Bfp8_b γ is TILE-only");
                // The slice's γ values: elements [e0, e0 + slice_elems) of the stick. The read starts at the
                // DRAM-read-aligned byte at or below e0 (a ROW_MAJOR shard slice need not start on a tile), so the
                // slice's first element lands delta bytes into the staging row (delta = 0 for tile-aligned slices).
                constexpr uint32_t GAMMA_READ_ALIGN = 64;  // >= every arch's DRAM read alignment
                const uint32_t first_byte = slice_e0 * es_g;
                const uint32_t aligned = first_byte & ~(GAMMA_READ_ALIGN - 1);
                const uint32_t stick_bytes = (slice_e0 + slice_elems) * es_g - aligned;
                if constexpr (gamma_mcast) {
                    [&](const auto& mc) {
                        if constexpr (std::decay_t<decltype(mc)>::active) {
                            if (mc.can_send()) {
                                auto sender = rms_norm_mcast::sender(mc, noc);
                                noc_async_read(gamma_accessor.get_noc_addr(0, aligned), gamma_row_addr, stick_bytes);
                                noc_async_read_barrier();
                                sender.send(gamma_row_addr, gamma_row_addr, stick_bytes);
                            } else {
                                auto receiver = rms_norm_mcast::receiver(mc, noc);
                                receiver.receive(0);
                            }
                        }
                    }(gmcast);
                } else {
                    noc_async_read(gamma_accessor.get_noc_addr(0, aligned), gamma_row_addr, stick_bytes);
                    noc_async_read_barrier();
                }
                gamma_row_addr += first_byte - aligned;
                gamma_chunks_filled = num_w_chunks;
            } else if constexpr (gamma_mcast && gamma_direct == GAMMA_DIRECT_OFF) {
                // TILE γ staging row: the injector fills every chunk up front and multicasts the row once.
                [&](const auto& mc) {
                    if constexpr (std::decay_t<decltype(mc)>::active) {
                        if (mc.can_send()) {
                            auto sender = rms_norm_mcast::sender(mc, noc);
                            for (uint32_t c = 0; c < num_w_chunks; ++c) {
                                fill_gamma_chunk(c);
                            }
                            noc_async_read_barrier();
                            sender.send(gamma_row_addr, gamma_row_addr, slice_valid * gamma_row_page);
                        } else {
                            auto receiver = rms_norm_mcast::receiver(mc, noc);
                            receiver.receive(0);
                        }
                    }
                }(gmcast);
                gamma_chunks_filled = num_w_chunks;
            }

            if (num_blocks > 0) {
                if constexpr (gamma_resident) {
                    // One reservation of the whole window, pushed chunk by chunk so compute can scale chunk 0
                    // as soon as it is ready (it waits cumulatively); the fill of chunk c+1 is in flight while
                    // chunk c is expanded.
                    cb_reserve_back(cb_gamma_tiles, Wp);
                    const uint32_t gamma_tiles_base = get_write_ptr(cb_gamma_tiles);
                    if (gamma_chunks_filled == 0) {
                        fill_gamma_chunk(0);
                    }
                    noc_async_read_barrier();
                    for (uint32_t c = 0; c < num_w_chunks; ++c) {
                        if (c + 1 < num_w_chunks && c + 1 >= gamma_chunks_filled) {
                            fill_gamma_chunk(c + 1);
                        }
                        const uint32_t valid = chunk_valid(c);
                        expand_gamma_tiles<gamma_row_page>(
                            gamma_row_addr, gamma_tiles_base + c * C * T_g, c * C, valid);
                        noc_async_read_barrier();
                        cb_push_back(cb_gamma_tiles, C);
                    }
                } else {
                    push_gamma_chunk(0);
                }
            }
        }
    }  // wr_gamma

    // Writer-side all-gather: CB bases captured before any push/pop (identical on every member).
    [[maybe_unused]] const uint32_t ag_inv_base = get_write_ptr(4);       // cb_inv_rms (never pushed here)
    [[maybe_unused]] const uint32_t ag_rms_root_base = get_read_ptr(11);  // cb_rms_root
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_first_row = block_idx * block_rows;
        const uint32_t remaining = rows_this_core - block_first_row;
        const uint32_t rows_in_block = remaining < block_rows ? remaining : block_rows;  // R_b

        for (uint32_t c = 0; c < num_w_chunks; ++c) {
            const bool is_tail_chunk = (c + 1 == num_w_chunks);
            if constexpr (has_gamma && !gamma_resident) {
                // One γ chunk ahead of the output drain (next step's chunk, wrapping into the next block).
                if (!(is_tail_chunk && block_idx + 1 == num_blocks)) {
                    push_gamma_chunk(is_tail_chunk ? 0 : c + 1);
                }
            }
            if constexpr (sharded) {
                if constexpr (ag_writer) {
                    if (c == 0) {  // once per block (this loop walks the W-chunks)
                        // (γ resident or absent: no γ push in this loop can wait behind the all-gather)
                        static_assert(!ag_writer || !has_gamma || gamma_resident, "AG_WRITER: resident γ only");
                        allgather_block<ag_G, ag_T_interm, block_rows, ag_wave_rows, ag_wave_q, ag_waves_full>(
                            ag_inv_sem_base,
                            ag_inv_base,
                            ag_rms_root_base,
                            ag_member_arg_base,
                            get_arg_val<uint32_t>(10),
                            rows_in_block);
                    }
                }
                continue;  // compute packs straight into the output shard
            }
            for (uint32_t r = 0; r < rows_in_block; ++r) {
                const uint32_t tile_row = start_tile_row + block_first_row + r;
                if constexpr (rm_sharded) {
                    // ROW_MAJOR shard: copy each valid stick's slice bytes of this chunk back into the core's
                    // own output shard (local L1, row stride shard_row_bytes). No TensorAccessor, no DRAM.
                    const uint32_t first_stick = tile_row * TILE_DIM;  // local to the shard
                    const uint32_t left = stick_count - first_stick;
                    const uint32_t sticks = left < TILE_DIM ? left : TILE_DIM;
                    const uint32_t first_byte = c * chunk_stick_bytes;
                    const uint32_t slice_bytes = slice_elems * es;
                    const uint32_t rest = slice_bytes > first_byte ? slice_bytes - first_byte : 0;
                    const uint32_t valid_bytes = rest < chunk_stick_bytes ? rest : chunk_stick_bytes;
                    cb_wait_front(cb_output, C);
                    if (valid_bytes > 0) {
                        const uint32_t l1 = get_read_ptr(cb_output);
                        const uint32_t dst = output_addr + first_stick * shard_row_bytes + first_byte;
                        for (uint32_t s = 0; s < sticks; ++s) {
                            noc_async_write(
                                l1 + s * chunk_stick_bytes, get_noc_addr(dst + s * shard_row_bytes), valid_bytes);
                        }
                        noc_async_writes_flushed();
                    }
                    cb_pop_front(cb_output, C);
                } else if constexpr (!is_rm) {
                    const uint32_t valid = chunk_valid(c);
                    {
                        MaybeDeviceZoneScope("wr_wait_out");
                        cb_wait_front(cb_output, C);
                    }
                    {
                        MaybeDeviceZoneScope("wr_issue");
                        const uint32_t l1 = get_read_ptr(cb_output);
                        const uint32_t first_page = tile_row * Wt + slice_w0 + c * C;
                        uint32_t j = stagger_first(bank_stagger, valid);
                        for (uint32_t n = 0; n < valid; ++n) {
                            if constexpr (!RMS_ABLATE_WRITES) {
                                noc_async_write_tile(first_page + j, output_accessor, l1 + j * T_out);
                            }
                            j = stagger_next(j, valid);
                        }
                        // Flush (data left L1, slot reusable) instead of a full ack barrier per quantum.
                        noc_async_writes_flushed();
                    }
                    cb_pop_front(cb_output, C);
                } else {
                    const uint32_t first_stick = tile_row * TILE_DIM;
                    const uint32_t left = num_sticks - first_stick;
                    const uint32_t sticks = left < TILE_DIM ? left : TILE_DIM;
                    if (tail_chunk_is_full || !is_tail_chunk) {
                        dataflow_kernel_lib::write_sticks_after_untilize<cb_output>(
                            output_accessor, sticks, chunk_stick_bytes, first_stick, c * chunk_stick_bytes);
                    } else {
                        cb_wait_front(cb_output, C);
                        const uint32_t l1 = get_read_ptr(cb_output);
                        for (uint32_t s = 0; s < sticks; ++s) {
                            noc_async_write(
                                l1 + s * chunk_stick_bytes,
                                output_accessor.get_noc_addr(first_stick + s, c * chunk_stick_bytes),
                                tail_stick_bytes);
                        }
                        noc_async_writes_flushed();
                        cb_pop_front(cb_output, C);
                    }
                }
            }
        }
    }
    MaybeDeviceZoneScope("wr_final_barrier");
    noc_async_write_barrier();
}

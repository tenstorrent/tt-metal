// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// argmax along the last dimension, on the data-movement harts (op_design.md).
//
// One DM kernel with `scan_threads` threads per cluster; thread t is the reader, the scanner and the writer
// of its own contiguous range of units [unit_start[t], unit_start[t] + unit_count[t]):
//   ROWS       (segmented = 0) a unit is one row; the thread writes the index of each row it finishes.
//   SEGMENTED  (segmented = 1) a unit is one (row, segment) of `spr` segments per row; the thread sends one
//              record {key, index, tag, 0} per unit to the finalizer's landing zone and raises the
//              finalizer's semaphore. The finalizer (thread 0 of the cluster holding unit 0) waits for all
//              records, merges each row's segments in column order and writes the indices.
//
// Block schedule (per thread):  load_block -> scan_block -> emit_block_results, IN_DEPTH blocks in flight;
// then send_partial (per finished unit, SEGMENTED), combine_partials + store_results (finalizer), release_noc.
//
// No kernel-library helper applies: the library's compute helpers act on tiles in DEST, and this op has no
// compute kernel (Quasar's NEOs have no index-carrying max). Every phase is raw NoC + scalar code.
//
// Comparison key (R1, R3): a bf16 word is sign-magnitude; key = sign ? -magnitude : magnitude, as int32.
// The key orders like the float value and maps -0.0 and +0.0 to the same key. A running best is replaced
// only on a strictly greater key, so the lowest index of a tie wins; the merge walks segments in column
// order with the same strict compare.
//
// DM cache (core_local_mem.h scoped_lock): every slot the NoC fills is invalidated before it is read, and
// every slot the CPU writes is flushed before a NoC write sources it. Slots one thread writes with the CPU
// are aligned to and spaced by a cache line (`line_bytes`), so no flush writes back another thread's bytes.
//
// Transaction ids (all user ids, < 8): reads read_trid_base + slot, output writes write_trid_base + slot,
// records record_trid. The last statements hand the hart's sticky tag back as id 0.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t T = get_arg(args::scan_threads);
constexpr uint32_t IN_DEPTH = get_arg(args::in_depth);
constexpr uint32_t OUT_DEPTH = get_arg(args::out_depth);
constexpr uint32_t READ_TRID = get_arg(args::read_trid_base);
constexpr uint32_t WRITE_TRID = get_arg(args::write_trid_base);
constexpr uint32_t RECORD_TRID = get_arg(args::record_trid);
constexpr uint32_t EB = get_arg(args::elem_bytes);
constexpr bool SEGMENTED = get_arg(args::segmented) != 0;
constexpr bool PROBE_IDS = get_arg(args::probe_ids) != 0;
constexpr uint32_t LINE = get_arg(args::line_bytes);
constexpr uint32_t RECORD_BYTES = get_arg(args::record_bytes);
constexpr uint32_t RECORD_WORDS = RECORD_BYTES / sizeof(uint32_t);

static_assert(EB == 2, "phase 0 scans bfloat16 only");
static_assert(RECORD_WORDS >= 3, "a record holds key, index and tag");
static_assert(IN_DEPTH >= 1 && OUT_DEPTH >= 1);

FORCE_INLINE int32_t bf16_key(uint32_t bits) {
    const int32_t mag = static_cast<int32_t>(bits & 0x7FFFu);
    return (bits & 0x8000u) ? -mag : mag;
}

FORCE_INLINE uint32_t align_up(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

// Running (key, index) of one row piece.
struct Best {
    int32_t key;
    uint32_t idx;
};

// Fold `n` bf16 elements at `p` (4-B aligned), whose first element is column `col0`, into `best`. With
// `init`, the running pair starts from the first element.
FORCE_INLINE void scan_piece(const uint16_t* p, uint32_t n, uint32_t col0, bool init, Best& best) {
    uint32_t j = 0;
    int32_t bk = best.key;
    uint32_t bi = best.idx;
    if (init) {
        bk = bf16_key(p[0]);
        bi = col0;
        j = 1;
        if (n > 1) {  // back onto a pair boundary
            const int32_t k = bf16_key(p[1]);
            if (k > bk) {
                bk = k;
                bi = col0 + 1;
            }
            j = 2;
        }
    }
    // Two elements per 32-bit load; the lower address (lower column) is the low half and goes first.
    const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
    for (; j + 1 < n; j += 2) {
        const uint32_t v = w[j >> 1];
        const int32_t k0 = bf16_key(v & 0xFFFFu);
        const int32_t k1 = bf16_key(v >> 16);
        if (k0 > bk) {
            bk = k0;
            bi = col0 + j;
        }
        if (k1 > bk) {
            bk = k1;
            bi = col0 + j + 1;
        }
    }
    if (j < n) {
        const int32_t k = bf16_key(p[j]);
        if (k > bk) {
            bk = k;
            bi = col0 + j;
        }
    }
    best.key = bk;
    best.idx = bi;
}

// One block: `n_units` row pieces starting at this thread's unit `unit0`. Whole pieces when the piece fits
// the block (col_off = 0, ncols = the piece's own width); otherwise one piece, columns [col_off, col_off+ncols).
struct Block {
    uint32_t unit0;
    uint32_t n_units;
    uint32_t col_off;
    uint32_t ncols;
};

}  // namespace

void kernel_main() {
    // ---- arguments ----
    const uint32_t t = get_my_thread_id();
    const uint32_t us = get_arg(args::unit_start)[t];
    const uint32_t uc = get_arg(args::unit_count)[t];
    const uint32_t is_finalizer = get_arg(args::is_finalizer);
    const uint32_t cluster_ordinal = get_arg(args::cluster_ordinal);

    const uint32_t tensor_w = get_arg(args::tensor_w);
    const uint32_t tensor_rows = get_arg(args::tensor_rows);
    const uint32_t spr = get_arg(args::spr);
    const uint32_t seg_cols = get_arg(args::seg_cols);
    const uint32_t block_bytes = get_arg(args::block_bytes);
    const uint32_t block_cols = get_arg(args::block_cols);
    const uint32_t block_rows = get_arg(args::block_rows);
    const uint32_t row_pitch = get_arg(args::row_pitch);
    const uint32_t stage_bytes = get_arg(args::stage_bytes);
    const uint32_t out_rows_per_page = get_arg(args::out_rows_per_page);
    const uint32_t num_records = get_arg(args::num_records);
    const uint32_t fin_x = get_arg(args::finalizer_noc_x);
    const uint32_t fin_y = get_arg(args::finalizer_noc_y);
    const uint32_t send_base = get_arg(args::send_base);
    const uint32_t send_stride = get_arg(args::send_stride);

    const bool whole_pieces = block_cols >= seg_cols;
    const uint32_t scanner_id = cluster_ordinal * T + t;  // probe tag

    Noc noc;
    const auto in_acc = TensorAccessor(tensor::input);
    const auto out_acc = TensorAccessor(tensor::output);
    Scratchpad<uint32_t> sp_blocks(scratch::sp_blocks);
    Scratchpad<uint32_t> sp_out(scratch::sp_out_stage);
    Scratchpad<uint32_t> sp_partials(scratch::sp_partials);

    // Each scratchpad's base aligned up to a cache line (the host padded each by one line). Offsets below
    // are relative to the scratchpad's own base, as the NoC calls take them.
    auto line_pad = [](const Scratchpad<uint32_t>& sp) {
        const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sp[0]));
        return align_up(base, LINE) - base;
    };
    auto l1_addr = [](const Scratchpad<uint32_t>& sp, uint32_t off) {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sp[0])) + off;
    };
    const uint32_t blocks_off = line_pad(sp_blocks) + t * IN_DEPTH * block_bytes;
    const uint32_t stage_off = line_pad(sp_out) + t * OUT_DEPTH * stage_bytes;
    const uint32_t partials_off = line_pad(sp_partials);
    const uint32_t send_off = partials_off + send_base + t * send_stride;

    // ---- unit geometry ----
    auto unit_row = [&](uint32_t u) { return u / spr; };
    auto unit_col_lo = [&](uint32_t u) { return (u % spr) * seg_cols; };
    auto unit_cols = [&](uint32_t u) {
        const uint32_t lo = unit_col_lo(u);
        return (tensor_w - lo) < seg_cols ? (tensor_w - lo) : seg_cols;
    };

    // ---- block cursor: walks this thread's units in blocks ----
    uint32_t cur_i = 0;  // next unit (relative to us)
    uint32_t cur_c = 0;  // next column within that unit's piece (split pieces only)
    auto next_block = [&](Block& b) -> bool {
        if (cur_i >= uc) {
            return false;
        }
        if (whole_pieces) {
            const uint32_t n = (uc - cur_i) < block_rows ? (uc - cur_i) : block_rows;
            b = Block{cur_i, n, 0, 0};
            cur_i += n;
        } else {
            const uint32_t pc = unit_cols(us + cur_i);
            const uint32_t n = (pc - cur_c) < block_cols ? (pc - cur_c) : block_cols;
            b = Block{cur_i, 1, cur_c, n};
            cur_c += n;
            if (cur_c >= pc) {
                ++cur_i;
                cur_c = 0;
            }
        }
        return true;
    };

    // load_block: one NoC read per row piece, landing at row_pitch strides in the thread's slot `slot`.
    auto load_block = [&](const Block& b, uint32_t slot) {
        const uint32_t slot_off = blocks_off + slot * block_bytes;
        for (uint32_t j = 0; j < b.n_units; ++j) {
            const uint32_t u = us + b.unit0 + j;
            const uint32_t ncols = whole_pieces ? unit_cols(u) : b.ncols;
            const uint32_t col = unit_col_lo(u) + b.col_off;
            noc.async_read<NocOptions::TXN_ID>(
                in_acc,
                sp_blocks,
                ncols * EB,
                {.page_id = unit_row(u), .offset_bytes = col * EB},
                {.offset_bytes = slot_off + j * row_pitch},
                {.trid = READ_TRID + slot});
        }
    };

    // Output: indices of rows [r0, r0 + n) staged at sp_out offset `src_off`; one write per output page.
    auto write_indices = [&](uint32_t src_off, uint32_t r0, uint32_t n, uint32_t trid) {
        while (n > 0) {
            const uint32_t page = r0 / out_rows_per_page;
            const uint32_t in_page = r0 % out_rows_per_page;
            const uint32_t m = (out_rows_per_page - in_page) < n ? (out_rows_per_page - in_page) : n;
            noc.async_write<NocOptions::TXN_ID>(
                sp_out,
                out_acc,
                m * sizeof(uint32_t),
                {.offset_bytes = src_off},
                {.page_id = page, .offset_bytes = in_page * static_cast<uint32_t>(sizeof(uint32_t))},
                {.trid = trid});
            src_off += m * sizeof(uint32_t);
            r0 += m;
            n -= m;
        }
    };

    Semaphore<> records_arrived(sem::records_arrived);

    // send_partial: one record to the finalizer's landing slot u, then the finalizer's semaphore up.
    auto send_partial = [&](uint32_t u, const Best& best) {
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = RECORD_TRID});  // send slot free again
        {
            CoreLocalMem<uint32_t> rec(l1_addr(sp_partials, send_off));
            auto lock = rec.scoped_lock(RECORD_WORDS);  // flushed on release
            rec[0] = static_cast<uint32_t>(best.key);
            rec[1] = best.idx;
            rec[2] = PROBE_IDS ? scanner_id + 1 : 0;
            rec[3] = 0;
        }
        noc.async_write<NocOptions::TXN_ID>(
            sp_partials,
            UnicastEndpoint{},
            RECORD_BYTES,
            {.offset_bytes = send_off},
            {.noc_x = fin_x, .noc_y = fin_y, .addr = l1_addr(sp_partials, partials_off + u * RECORD_BYTES)},
            {.trid = RECORD_TRID});
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = RECORD_TRID});  // landed before it is counted
        records_arrived.up(noc, fin_x, fin_y, 1);
    };

    // ---- the block schedule ----
    Block pending[IN_DEPTH];
    bool has[IN_DEPTH];
    for (uint32_t s = 0; s < IN_DEPTH; ++s) {
        has[s] = next_block(pending[s]);
        if (has[s]) {
            load_block(pending[s], s);
        }
    }

    Best running{0, 0};  // carried across the column blocks of a split piece
    uint32_t emits = 0;
    for (uint32_t k = 0;; ++k) {
        const uint32_t s = k % IN_DEPTH;
        if (!has[s]) {
            break;
        }
        const Block b = pending[s];
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = READ_TRID + s});

        const uint32_t o = emits % OUT_DEPTH;
        const uint32_t o_off = stage_off + o * stage_bytes;
        uint32_t n_done = 0;
        uint32_t first_row = 0;
        if constexpr (!SEGMENTED) {
            noc.async_write_barrier<NocOptions::TXN_ID>({.trid = WRITE_TRID + o});  // staging slot free again
        }
        {
            // scan_block: invalidate the slot, fold every piece into its running pair.
            const uint32_t slot_off = blocks_off + s * block_bytes;
            CoreLocalMem<uint32_t> slot(l1_addr(sp_blocks, slot_off));
            auto in_lock = slot.scoped_lock(b.n_units * row_pitch / sizeof(uint32_t));
            CoreLocalMem<uint32_t> stage(l1_addr(sp_out, o_off));
            auto out_lock = stage.scoped_lock(SEGMENTED ? 0 : stage_bytes / sizeof(uint32_t));
            const uint16_t* slot_p = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(l1_addr(sp_blocks, slot_off)));

            for (uint32_t j = 0; j < b.n_units; ++j) {
                const uint32_t u = us + b.unit0 + j;
                const uint32_t piece_cols = unit_cols(u);
                const uint32_t ncols = whole_pieces ? piece_cols : b.ncols;
                const bool first = b.col_off == 0;
                const bool last = b.col_off + ncols == piece_cols;
                scan_piece(
                    slot_p + j * (row_pitch / EB), ncols, unit_col_lo(u) + b.col_off, first, running);
                if (!last) {
                    continue;
                }
                // emit_block_results (ROWS) / send_partial (SEGMENTED)
                if constexpr (SEGMENTED) {
                    send_partial(u, running);
                } else {
                    if (n_done == 0) {
                        first_row = unit_row(u);
                    }
                    stage[n_done++] = PROBE_IDS ? scanner_id : running.idx;
                }
            }
        }  // flushes the staging slot
        if constexpr (!SEGMENTED) {
            if (n_done > 0) {
                write_indices(o_off, first_row, n_done, WRITE_TRID + o);
                ++emits;
            }
        }
        has[s] = next_block(pending[s]);
        if (has[s]) {
            load_block(pending[s], s);
        }
    }

    // ---- combine_partials + store_results (SEGMENTED, finalizer thread only) ----
    if constexpr (SEGMENTED) {
        if (is_finalizer && t == 0) {
            records_arrived.wait(num_records);
            records_arrived.set(0);  // ready for the next run of this program
            const uint32_t o_off = stage_off;  // slot 0 of thread 0 holds tensor_rows indices
            noc.async_write_barrier<NocOptions::TXN_ID>({.trid = WRITE_TRID});
            {
                CoreLocalMem<uint32_t> zone(l1_addr(sp_partials, partials_off));
                auto zone_lock = zone.scoped_lock(num_records * RECORD_WORDS);
                CoreLocalMem<uint32_t> stage(l1_addr(sp_out, o_off));
                auto out_lock = stage.scoped_lock(stage_bytes / sizeof(uint32_t));
                for (uint32_t r = 0; r < tensor_rows; ++r) {
                    const uint32_t base = r * spr * RECORD_WORDS;
                    int32_t bk = static_cast<int32_t>(zone[base]);
                    uint32_t bi = zone[base + 1];
                    uint32_t tag_sum = zone[base + 2];
                    for (uint32_t sg = 1; sg < spr; ++sg) {
                        const uint32_t rec = base + sg * RECORD_WORDS;
                        const int32_t k = static_cast<int32_t>(zone[rec]);
                        if (k > bk) {
                            bk = k;
                            bi = zone[rec + 1];
                        }
                        tag_sum += zone[rec + 2];
                    }
                    stage[r] = PROBE_IDS ? tag_sum : bi;
                }
            }  // flushes the staged indices
            write_indices(o_off, 0, tensor_rows, WRITE_TRID);
        }
    }

    // ---- release_noc ----
    for (uint32_t o = 0; o < OUT_DEPTH; ++o) {
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = WRITE_TRID + o});
    }
    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = RECORD_TRID});
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc_async_write_set_trid(0, noc.get_noc_id());
}

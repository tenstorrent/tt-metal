// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF kernel (`split_reader_v2`) — the DATA MOVEMENT role, for
// BOTH data-movement RISCs. One source file so the baseline arm and every
// candidate arm differ ONLY in compile-time args; there is no second code path
// that could quietly become the delta.
//
// Carried over from Perf-1's `split_reader/experiment_kernels/sr_dm.cpp` and
// extended with ONE new split mode (`split == 3`, the PERIODIC weighted
// interleave) — see the note on `period` / `share` below for why the Perf-1
// contiguous weighted split cannot be integrated as measured.
//
// Roles, chosen per RISC by the host:
//
//   do_read  0 — this RISC issues no reads
//            1 — this RISC reads the subset of blocks selected by `split`/`phase`
//                into `cb_in`
//   do_write 0 — this RISC issues no writes and drains nothing
//            1 — TensorAccessor write of whole output TILE pages (interleaved
//                destination), one barrier per block
//            2 — DRAIN only (destination-local: the output CB is aliased on the
//                resident shard, so the "writer" only pops pages)
//   split    0 — this RISC owns ALL num_blocks blocks (the op's current scheme)
//            1 — stride-2 interleave: it owns blocks {phase, phase+2, ...}
//            2 — contiguous half: phase 0 owns [0, n0), phase 1 owns [n0, n)
//            3 — PERIODIC weighted interleave with period `period`: phase 0 owns
//                the blocks with (i % period) < share, phase 1 the rest.
//
// Why mode 3 exists (the Perf-2 finding that mode 2 cannot ship): a CONTIGUOUS
// split needs the second RISC's CB to be as deep as its whole half, because
// compute does not touch the second half until it has drained the first — cap
// that depth and the two halves serialize, which is the whole win gone. Its L1
// therefore scales with BLOCKS-PER-CORE, i.e. with the tensor, which is exactly
// what the op's CB contract forbids. Mode 3 delivers the same 3:1 issue ratio
// with a BOUNDED CB depth (the run-ahead is `share`, not `n0`).
//
// A block is 1 tile-row x `wt_chunk` tile-columns, exactly as the real op
// (op_design.md §1). Two work assignments, same as the op: W_REGION (the core's
// own destination-shard tile band, one W chunk per row) and W_BLOCKS (a
// contiguous range of the global W-chunk-major block index).
//
// HELPER USAGE / SUBSTITUTION (reported to the coordinator):
//   `use_helper == 1` routes every aligned read through
//   dataflow_kernel_lib::read_sticks_for_tilize<TILE>, which is what the op does
//   today — so the baseline arm is the library call verbatim, and the candidate
//   keeps using it (one call per contiguous run; one call per block when the
//   split is an interleave, since an interleave has no run). The helper barriers
//   PER BLOCK internally either way (tilize_helpers_dataflow.inl:126), so
//   splitting one N-block call into N one-block calls changes no transfer and no
//   barrier policy — only the per-call prologue is paid N times.
//   `use_helper == 0` is the raw-issue arm, needed for the per-RISC transaction
//   id (the helper owns its barrier and exposes no trid).
//
// The gather regime (`regime == 1`) is the op's R_PAD branch minus the pad fill:
// a source shard NARROWER than a tensor row makes one row several pages, which
// read_sticks_for_tilize's contract (page id == stick index) cannot address. The
// op substitutes the helper there for the same reason; this bench inherits it.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {

constexpr uint32_t TILE_W = 32;

// One row-major source row's span, byte-identical to the op's read_row_span():
// one transfer when a page IS a stick, one transfer per page slice when the
// source shard is narrower than the row.
template <uint32_t page_bytes, uint32_t row_pages, typename Accessor>
FORCE_INLINE void read_row_span(
    const Accessor& accessor, uint32_t row, uint32_t byte_off, uint32_t n_bytes, uint32_t l1_addr) {
    if constexpr (row_pages == 1) {
        noc_async_read(accessor.get_noc_addr(row, byte_off), l1_addr, n_bytes);
    } else {
        uint32_t page_in_row = byte_off / page_bytes;
        uint32_t page = row * row_pages + page_in_row;
        uint32_t off = byte_off - page_in_row * page_bytes;
        while (n_bytes > 0) {
            uint32_t n = page_bytes - off;
            if (n > n_bytes) {
                n = n_bytes;
            }
            noc_async_read(accessor.get_noc_addr(page, off), l1_addr, n);
            l1_addr += n;
            n_bytes -= n;
            ++page;
            off = 0;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t do_read = get_compile_time_arg_val(0);
    constexpr uint32_t do_write = get_compile_time_arg_val(1);
    constexpr uint32_t regime = get_compile_time_arg_val(2);  // 0 = aligned sticks, 1 = paged gather
    constexpr uint32_t cb_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t work_mode = get_compile_time_arg_val(5);  // 0 = W_BLOCKS, 1 = W_REGION
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(7);
    constexpr uint32_t nt_h = get_compile_time_arg_val(8);
    constexpr uint32_t wt = get_compile_time_arg_val(9);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t src_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t src_row_pages = get_compile_time_arg_val(12);
    constexpr uint32_t split = get_compile_time_arg_val(13);
    constexpr uint32_t phase = get_compile_time_arg_val(14);
    constexpr uint32_t use_helper = get_compile_time_arg_val(15);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(16);
    // Per-RISC read TRANSACTION ID (0 = off). Two RISCs sharing ONE NoC cannot
    // use the plain read barrier: in DEDICATED mode it compares the NIU's
    // hardware response counter against the issuing RISC's own local counter,
    // and in DYNAMIC mode it compares against the SUM of both RISCs' counters —
    // so each reader would wait for the other's outstanding reads. A trid
    // barrier checks NIU_MST_REQS_OUTSTANDING_ID(trid), a per-transaction-id
    // hardware counter that is RISC-agnostic, so distinct trids let two readers
    // share one NoC and still barrier only on their OWN reads.
    //
    // HELPER SUBSTITUTION (justified, `capability`): read_sticks_for_tilize owns
    // its barrier internally and its contract exposes no transaction id, so the
    // trid arm has to issue raw (the host forces use_helper = 0 with it). Same
    // transfers, same one-barrier-per-block policy — only the barrier's SCOPE
    // changes. This is the same substitution the real op already documents for
    // master.md B8.
    constexpr uint32_t trid = get_compile_time_arg_val(17);
    // split == 3 only: the weighting period and this-RISC-phase-0's share of it.
    constexpr uint32_t period = get_compile_time_arg_val(18);
    constexpr uint32_t share = get_compile_time_arg_val(19);
    constexpr auto src_args = TensorAccessorArgs<20>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t row_bytes = wt_chunk * TILE_W * elem_bytes;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(4);
    const uint32_t col_off_base = get_arg_val<uint32_t>(5);
    const uint32_t n0 = get_arg_val<uint32_t>(6);  // contiguous-half boundary

    if (num_blocks == 0) {
        return;
    }

    [[maybe_unused]] const auto src = TensorAccessor(src_args, src_addr);
    [[maybe_unused]] const auto dst = TensorAccessor(dst_args, dst_addr);

    // ── block index -> geometry (identical map on both RISCs) ───────────────
    [[maybe_unused]] auto tile_row_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1) {
            return tile_row0 + i;  // W_REGION, one W chunk per shard row
        } else {
            const uint32_t b = start_block + i;
            return b % nt_h;  // W-chunk-major
        }
    };
    [[maybe_unused]] auto wchunk_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1) {
            return 0;
        } else {
            const uint32_t b = start_block + i;
            return b / nt_h;
        }
    };
    [[maybe_unused]] auto byte_off_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1) {
            return col_off_base;
        } else {
            return wchunk_of(i) * row_bytes;
        }
    };

    // ── does THIS RISC own block i? ─────────────────────────────────────────
    [[maybe_unused]] auto owns = [&](uint32_t i) -> bool {
        if constexpr (split == 0) {
            return true;
        } else if constexpr (split == 1) {
            return (i & 1u) == phase;
        } else if constexpr (split == 2) {
            return phase ? (i >= n0) : (i < n0);
        } else {
            return ((i % period) < share) == (phase == 0);
        }
    };

    // The first index this RISC could own — `owns()` still filters, this only
    // skips a prefix it provably does not own.
    [[maybe_unused]] const uint32_t first_owned = (split == 1) ? phase : ((split == 2 && phase) ? n0 : 0u);
    [[maybe_unused]] const uint32_t end_owned = (split == 2 && phase == 0) ? n0 : num_blocks;

    // ── ONE block's read ────────────────────────────────────────────────────
    [[maybe_unused]] auto read_block = [&](uint32_t i) {
        const uint32_t first_row = tile_row_of(i) * tile_h;
        const uint32_t byte_off = byte_off_of(i);
        if constexpr (regime == 0 && use_helper) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
                src, tile_h, row_bytes, first_row, byte_off);
        } else {
            cb_reserve_back(cb_in, wt_chunk);
            uint32_t l1_addr = get_write_ptr(cb_in);
            for (uint32_t r = 0; r < tile_h; ++r) {
                if constexpr (regime == 0) {
                    noc_async_read(src.get_noc_addr(first_row + r, byte_off), l1_addr, row_bytes);
                } else {
                    read_row_span<src_page_bytes, src_row_pages>(src, first_row + r, byte_off, row_bytes, l1_addr);
                }
                l1_addr += row_bytes;
            }
            if constexpr (trid) {
                noc_async_read_barrier_with_trid(trid);
            } else {
                noc_async_read_barrier();
            }
            cb_push_back(cb_in, wt_chunk);
        }
    };

    // ── a contiguous RUN of blocks as ONE helper call ───────────────────────
    // Consecutive blocks of one run share a W chunk and march up the tile-rows,
    // so run*tile_h source sticks are consecutive page ids at one byte offset —
    // exactly read_sticks_for_tilize's contract. This is the op's hot path.
    [[maybe_unused]] auto read_run = [&](uint32_t first, uint32_t count) {
        if constexpr (regime == 0 && use_helper) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
                src, count * tile_h, row_bytes, tile_row_of(first) * tile_h, byte_off_of(first));
        } else {
            for (uint32_t i = first; i < first + count; ++i) {
                read_block(i);
            }
        }
    };

    // ── ONE block's write / drain ───────────────────────────────────────────
    [[maybe_unused]] auto write_block = [&](uint32_t i) {
        cb_wait_front(cb_out, wt_chunk);
        if constexpr (do_write == 1) {
            uint32_t l1_addr = get_read_ptr(cb_out);
            const uint32_t first_page = tile_row_of(i) * wt + wchunk_of(i) * wt_chunk;
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                noc_async_write(l1_addr, dst.get_noc_addr(first_page + k), out_tile_bytes);
                l1_addr += out_tile_bytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, wt_chunk);
    };

    // Tag every read this RISC issues with its own transaction id.
    if constexpr (do_read && trid) {
        noc_async_read_set_trid(trid);
    }

    // ── role dispatch ───────────────────────────────────────────────────────
    if constexpr (do_read && !do_write) {
        if constexpr (split == 1 || split == 3) {
            // An interleave has no contiguous run, so one call per owned block.
            for (uint32_t i = first_owned; i < num_blocks; ++i) {
                if (owns(i)) {
                    read_block(i);
                }
            }
        } else if constexpr (work_mode == 1) {
            // W_REGION: this RISC's whole subset is ONE run of tile-rows.
            read_run(first_owned, end_owned - first_owned);
        } else {
            // W_BLOCKS: split the subset into runs that share a W chunk.
            uint32_t i = first_owned;
            while (i < end_owned) {
                uint32_t run = nt_h - tile_row_of(i);
                if (run > end_owned - i) {
                    run = end_owned - i;
                }
                read_run(i, run);
                i += run;
            }
        }
    } else if constexpr (!do_read && do_write) {
        for (uint32_t i = 0; i < num_blocks; ++i) {
            write_block(i);
        }
    } else if constexpr (do_read && do_write) {
        // BOTH roles on one RISC — the arm for an interleaved DESTINATION (where
        // the writer is NOT free) and for a destination-local plan that still
        // needs its writer for the pad stamp (do_write == 2, drain only). One
        // block of read lookahead so this RISC's read duty is not strictly
        // serialized behind its own write duty.
        uint32_t next_read = first_owned;
        auto pump = [&](uint32_t upto) {
            while (next_read < num_blocks && next_read <= upto) {
                if (owns(next_read)) {
                    read_block(next_read);
                }
                ++next_read;
            }
        };
        pump(0);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            pump(i + 1);
            write_block(i);
        }
    }

    if constexpr (do_read && trid) {
        noc_async_read_set_trid(0);  // leave the command buffer's tag clean
    }
}

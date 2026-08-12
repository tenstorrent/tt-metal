// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader — NCRISC / NoC0.
//
// One block = one output tile-row x `w` output tile-columns. Blocks are
// linearized `b = wchunk * nt_h + r`; this core owns the contiguous range
// [b0, b0 + nb). Per block the reader issues `tile_h` NoC reads of
// `w * tile_row_bytes` bytes at L1 stride `w * tile_row_bytes` and ONE barrier,
// then pushes `w` tile-sized pages — exactly the contract
// `dataflow_kernel_lib::read_sticks_for_tilize<TILE>` implements, which is why
// the production path is that helper call and nothing else.
//
// `resident == 1` (Refinement 2, A3/C14) is the SECOND production path and the
// only one that is not a NoC read at all: the input shard already sits in this
// core's L1 and `cb_input_sticks` is aliased onto it, so the reader just arms
// the CB. That is what implementing sharding means — re-reading a local shard
// through the TensorAccessor above would re-fetch bytes the core already holds.
//
// The `if constexpr` arms below are LEVER COUNTERFACTUALS, not alternative
// production paths: `barrier_per_block == 0` is the master.md B7 off-arm (one
// barrier per transaction instead of per block) and `stub_read == 1` is the
// /perf-measure read ablation (keep the CB sync scaffolding, drop the NoC
// payload). At their defaults (1, 0) the compiler emits only the helper call, so
// they cannot perturb the measured path.
//
// `n_bands > 1` (Refinement 4, A3c) is the THIRD production path: the source is
// sharded narrower than a row, so its pages are BANDS of a row scattered across
// cores, and this core (which owns the OUTPUT block) pulls the bands it needs
// over L1->L1. See the arm's own comment for the topology.
//
// HELPER SUBSTITUTION, declared (Refinement 4): the gather arm cannot be
// `read_sticks_for_tilize` either. That helper reads ONE page per stick at a
// fixed `byte_offset_within_page` (`tilize_helpers_dataflow.inl`), which is the
// whole-row-page contract; a banded source needs the stick's bytes assembled
// from SEVERAL pages at different offsets, into one contiguous L1 row. There is
// no kernel_lib entry point for a segmented stick read, so `noc_async_read` +
// `TensorAccessor::get_noc_addr(page, offset)` is the only mechanism — and the
// helper remains the emitted code on the whole-row path (`n_bands == 1`), which
// is every pre-R4 shape, byte for byte.
//
// `pad_enabled == 1` (Refinement 5, P1/P2/P4/P5) is the FOURTH production path:
// the output's tile grid is a PAD TARGET larger than the input, so a block can
// contain positions no input byte maps to. It is a CT-selected second body —
// at `pad_enabled == 0` the compiler emits none of it, which is what makes
// "the aligned path is unchanged" structural rather than a convention.
//
// HELPER SUBSTITUTION, declared (Refinement 5): only the BOUNDARY blocks of the
// pad body are hand-written; a fully-interior block still calls
// `read_sticks_for_tilize` (see the arm below). Neither of the two kernel_lib
// candidates can serve a boundary block. `l1_helpers.hpp::zero_tile` writes only
// ZEROS through the NoC's write-zeros engine, and the whole point of the
// `pad_value` sign buckets is an ARBITRARY fill. `read_sticks_for_tilize` leaves
// the pad region as STALE L1 (its own doc says so: "untouched rows contain stale
// data"), and it derives its L1 row stride from the bytes it is asked to read —
// `round_up(row_bytes, tile_row_bytes)` — which is the block width only when the
// block ends at the real data; a block that extends into whole pad tile-COLUMNS
// (`[1,1,32,50] -> [1,1,32,128]`) would get a 2-tile stride for a 4-tile block
// and scatter the sticks. So the fill is a plain L1 store loop (no NoC traffic
// at all) and the clamped read is `noc_async_read` + `get_noc_addr(page, off)`.
//
// HELPER SUBSTITUTION, declared (Refinement 6): `fast_addrgen == 1` (master.md
// D21) and `stateful_reads == 1` (master.md B13) issue the SAME `tile_h` reads
// of the SAME size to the SAME L1 destinations — they only make each issue
// cheaper: D21 replaces `TensorAccessor::get_noc_addr`'s two divisions by the
// bank count with one addition per stick, B13 replaces six command-buffer
// register writes with three. `read_sticks_for_tilize` cannot express either: it
// calls `noc_async_read` with an accessor address per stick and exposes no seam
// for a caller-maintained address table or a stateful command buffer. The issue
// ORDER is unchanged (that was measured — see the arm), so the substitution is
// the per-issue cost and nothing else, and the helper remains the emitted code
// whenever both levers are off.
//
// HELPER SUBSTITUTION, declared (Refinement 3): `stagger_reads == 1` issues the
// same `tile_h` reads to the same L1 destinations in a ROTATED order, and
// `read_sticks_for_tilize` cannot express that — it walks `start_page ..
// start_page + total_num_rows` sequentially into consecutive L1, so the only way
// to reorder the issue is to split it into two calls, which would write the
// sticks into the CB in rotated order and produce a row-permuted tile. The
// substitution is therefore the rotation itself, not a preference; it reuses the
// arm that already existed for B7, and the helper remains the emitted code
// whenever the lever is off (the `stagger_reads == 0` condition above).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

// Fill `bytes` of L1 at `addr` with `pad_word` (op_design.md §8.3's fill).
//
// `pad_word` is the fill value already replicated across the 32-bit word by the
// host (`pad_fill_word`), so the fast path is a word-store loop. The unaligned
// head/tail exist because a region can start at the real row WIDTH, which is a
// multiple of the element size but need not be a multiple of 4 (e.g. an odd W
// at bf16, or any W at uint8). The element size always DIVIDES 4, so the word
// repeats with a period that divides 4 and byte `A` of the fill is
// `pad_word >> ((A & 3) * 8)` — the phase is carried by the address itself.
FORCE_INLINE void fill_pad_region(uint32_t addr, uint32_t bytes, uint32_t pad_word) {
    const uint32_t end = addr + bytes;
    while (addr < end && (addr & 3u) != 0u) {
        *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr) = (pad_word >> ((addr & 3u) * 8)) & 0xFFu;
        ++addr;
    }
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t n_words = (end - addr) >> 2;
    for (uint32_t i = 0; i < n_words; ++i) {
        words[i] = pad_word;
    }
    addr += n_words << 2;
    while (addr < end) {
        *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr) = (pad_word >> ((addr & 3u) * 8)) & 0xFFu;
        ++addr;
    }
}

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t nt_h = get_compile_time_arg_val(1);             // tile-rows
    constexpr uint32_t n_wchunks = get_compile_time_arg_val(2);        // column-blocks per tile-row
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);           // sticks per tile-row
    constexpr uint32_t tile_row_bytes = get_compile_time_arg_val(4);   // 32 * elem
    constexpr uint32_t wt_block = get_compile_time_arg_val(5);         // the block-width knob
    constexpr uint32_t wt_tail = get_compile_time_arg_val(6);
    constexpr uint32_t barrier_per_block = get_compile_time_arg_val(7);  // lever B7 (1 = on)
    constexpr uint32_t stub_read = get_compile_time_arg_val(8);          // ablation (0 = off)
    constexpr uint32_t resident = get_compile_time_arg_val(9);           // A3/C14 zero-copy (1 = on)
    constexpr uint32_t stagger_reads = get_compile_time_arg_val(10);     // lever R3/A3 (1 = on)
    constexpr uint32_t n_bands = get_compile_time_arg_val(11);           // R4: source pages per row
    constexpr uint32_t band_bytes = get_compile_time_arg_val(12);        // R4: bytes per source page
    constexpr uint32_t pad_enabled = get_compile_time_arg_val(13);       // R5: the pad body (1 = on)
    constexpr uint32_t pad_word = get_compile_time_arg_val(14);          // R5: fill, input format, replicated
    constexpr uint32_t pad_hp = get_compile_time_arg_val(15);            // R5: PADDED rows per image
    constexpr uint32_t pad_h_real = get_compile_time_arg_val(16);        // R5: REAL rows per image
    constexpr uint32_t pad_nimg = get_compile_time_arg_val(17);          // R5: REAL images
    constexpr uint32_t pad_row_bytes = get_compile_time_arg_val(18);     // R5: REAL bytes per stick
    constexpr uint32_t stateful_reads = get_compile_time_arg_val(19);    // lever R6/B13 (1 = on)
    constexpr uint32_t bank_period = get_compile_time_arg_val(20);       // R6: source pages per bank cycle
    constexpr uint32_t fast_addrgen = get_compile_time_arg_val(21);      // lever R6/D21 (1 = on)
    constexpr auto src_args = TensorAccessorArgs<22>();

    // The largest transfer this kernel can issue. `one_packet` NoC state is only
    // legal at or below the part's burst size, so the arch decides which
    // stateful primitive pair the B13 arm uses — never a hardcoded byte count.
    constexpr bool stateful_one_packet = (wt_block * tile_row_bytes) <= NOC_MAX_BURST_SIZE;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1);
    const uint32_t nb = get_arg_val<uint32_t>(2);

    // A3/C14 zero-copy: `cb_input_sticks` is ALIASED onto this core's own input
    // shard, so the block is already in L1 and there is nothing to fetch — the
    // reader exists only to arm the CB. NO NoC read, no TensorAccessor, and the
    // shard hands us the block width (`wt_block == Wt_shard`), so the whole
    // shard is `nb * wt_block` tile-sized pages.
    if constexpr (resident == 1) {
        const uint32_t pages = nb * wt_block;
        cb_reserve_back(cb_input_sticks, pages);
        cb_push_back(cb_input_sticks, pages);
        return;
    }

    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < nb; ++i) {
        const uint32_t b = b0 + i;
        const uint32_t wchunk = b / nt_h;      // column-block index
        const uint32_t r = b - wchunk * nt_h;  // global tile-row index

        // The tail column-block is the last one; its width is WT_TAIL (== WT_BLOCK
        // when Wt divides evenly), so the reader's per-block page count matches
        // compute's `WT_BLOCK x n_full` then `WT_TAIL x n_tail` sequence exactly.
        const uint32_t w = (wchunk == n_wchunks - 1) ? wt_tail : wt_block;
        const uint32_t row_bytes = w * tile_row_bytes;
        const uint32_t byte_offset = wchunk * wt_block * tile_row_bytes;

        if constexpr (pad_enabled == 1) {
            // R5 (P1/P2/P4/P5) PADDED READER — one body, all three pad regions.
            //
            // The block's `tile_h` output rows are the padded rows
            // `[g0, g0+tile_h)` of the padded grid. Because `pad_hp` is a whole
            // number of tile-rows, every row of a block belongs to the SAME
            // image, so the real/pad split is two scalars per block:
            //   real_rows  — rows of this block that exist in the input
            //                (0 when the block is a whole pad tile-ROW, or when
            //                 the image itself is past the input's batch)
            //   real_bytes — bytes of a real row that exist in the input
            //                (0 when the block is a whole pad tile-COLUMN)
            // Their three complements ARE the three pad regions: the H tail and
            // whole pad tile-rows are `rows >= real_rows`; the W tail and whole
            // pad tile-columns are `bytes >= real_bytes` of a real row.
            const uint32_t g0 = r * tile_h;
            const uint32_t img = g0 / pad_hp;
            const uint32_t row0 = g0 - img * pad_hp;
            uint32_t real_rows = 0;
            if (img < pad_nimg && row0 < pad_h_real) {
                real_rows = pad_h_real - row0;
                if (real_rows > tile_h) {
                    real_rows = tile_h;
                }
            }
            uint32_t real_bytes = 0;
            if (byte_offset < pad_row_bytes) {
                real_bytes = pad_row_bytes - byte_offset;
                if (real_bytes > row_bytes) {
                    real_bytes = row_bytes;
                }
            }
            // Source stick of output row `g0+s`: the input's rows are dense, so
            // the padded row index has to be projected back onto them.
            const uint32_t start_page = img * pad_h_real + row0;

            if (real_rows == tile_h && real_bytes == row_bytes) {
                // Fully interior block — no pad position in it at all, so it is
                // the aligned path verbatim, helper and all. This is the common
                // case (a pad touches only the last tile-row / tile-column), and
                // it is why a padded call costs the fill only where it must.
                dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                    src,
                    /*total_num_rows*/ tile_h,
                    /*row_bytes*/ row_bytes,
                    /*start_page*/ start_page,
                    /*byte_offset_within_page*/ byte_offset);
            } else {
                cb_reserve_back(cb_input_sticks, w);
                const uint32_t l1_base = get_write_ptr(cb_input_sticks);
                // Region A — whole pad rows (H tail + whole pad tile-rows).
                fill_pad_region(l1_base + real_rows * row_bytes, (tile_h - real_rows) * row_bytes, pad_word);
                // Region B — the tail of each real row (W tail + whole pad
                // tile-columns, where `real_bytes == 0` and the row is all fill).
                if (real_bytes < row_bytes) {
                    for (uint32_t s = 0; s < real_rows; ++s) {
                        fill_pad_region(l1_base + s * row_bytes + real_bytes, row_bytes - real_bytes, pad_word);
                    }
                }
                // The real sub-rectangle, read into the region the fill left for
                // it — DISJOINT from both fills, so a store can never land on a
                // byte an in-flight read owns. Still ONE barrier per block (B7).
                if (real_bytes > 0) {
                    for (uint32_t s = 0; s < real_rows; ++s) {
                        if constexpr (stub_read == 0) {
                            noc_async_read(
                                src.get_noc_addr(start_page + s, byte_offset), l1_base + s * row_bytes, real_bytes);
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_sticks, w);
            }
        } else if constexpr (n_bands > 1) {
            // R4 (A3c) CROSS-SPEC RESHARD — the PULL gather.
            //
            // The source is sharded NARROWER than a row, so its pages are bands:
            // page `(row, band)` holds bytes [band*band_bytes, +band_bytes) of
            // that row and lives in whatever core's L1 owns that shard. This
            // core owns the OUTPUT block, so it pulls the bands its block needs
            // — the read is split at band boundaries and each segment is one
            // NoC read from whichever core holds it. Still ONE barrier per block
            // (B7), still no semaphore and no multicast: §1.1 makes the map a
            // bijection, so every source byte is read by exactly one core and
            // there is nothing to coordinate.
            //
            // `band_bytes` is a whole number of tile-columns (the host refuses
            // anything else), so every segment length is a multiple of
            // `tile_row_bytes` and the L1 cursor stays on the alignment grid.
            cb_reserve_back(cb_input_sticks, w);
            const uint32_t l1_base = get_write_ptr(cb_input_sticks);
            const uint32_t end = byte_offset + row_bytes;
            for (uint32_t s = 0; s < tile_h; ++s) {
                const uint32_t row_page0 = (r * tile_h + s) * n_bands;
                uint32_t l1 = l1_base + s * row_bytes;
                uint32_t off = byte_offset;
                while (off < end) {
                    const uint32_t band = off / band_bytes;
                    const uint32_t in_band = off - band * band_bytes;
                    uint32_t len = band_bytes - in_band;
                    if (len > end - off) {
                        len = end - off;
                    }
                    if constexpr (stub_read == 0) {
                        noc_async_read(src.get_noc_addr(row_page0 + band, in_band), l1, len);
                        if constexpr (barrier_per_block == 0) {
                            noc_async_read_barrier();  // B7 off: one barrier per transaction
                        }
                    }
                    l1 += len;
                    off += len;
                }
            }
            if constexpr (barrier_per_block == 1) {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, w);
        } else if constexpr ((stateful_reads == 1 || fast_addrgen == 1) && barrier_per_block == 1) {
            // R6 CHEAPER READ ISSUE — master.md D21 (`fast_addrgen`) and B13
            // (`stateful_reads`). SAME transfers, SAME sizes, SAME L1
            // destinations, and — the part that is load-bearing — the SAME issue
            // ORDER as the helper: stick 0, 1, ... `tile_h - 1`.
            //
            // The low-work regimes are ISSUE-bound, not bandwidth-bound
            // (`[1,1,32,64]` moves 4 KB in 32 reads at ~30 ns each), so what is
            // worth attacking is the per-read WORK. The largest piece of it is
            // address generation: `TensorAccessor::get_noc_addr` splits the page
            // into (bank, page-in-bank) with two divisions by the bank count, and
            // this part has SEVEN banks — not a power of two, so both are
            // software divides.
            //
            // D21 removes them without touching the order. An interleaved buffer
            // maps page p to bank `p % bank_period` at page `p / bank_period`
            // inside it, so for a FIXED bank phase the addresses form an
            // arithmetic progression of one aligned page. `bank_addr[]` therefore
            // carries one running address per phase, walked in natural stick
            // order with a wrapping counter (never a modulo): `bank_period + 1`
            // accessor calls per block instead of `tile_h`.
            //
            // Issuing in bank-phase order instead — which would let ONE
            // `set_state` cover a whole group — was built and measured, and it is
            // 1.18-1.27x SLOWER: five consecutive requests to the same bank queue
            // behind each other, while the natural order round-robins the banks.
            // So B13 keeps the natural order too, where the source node changes
            // every stick and the state has to be re-programmed every stick; it
            // stays a live knob (default off) with that measurement recorded.
            //
            // `bank_period` is a pure PERFORMANCE hint, never a correctness
            // assumption: the stride is DERIVED from two real accessor addresses
            // and used only if they agree on the source node (`affine`),
            // otherwise every stick falls back to the accessor.
            cb_reserve_back(cb_input_sticks, w);
            const uint32_t l1_base = get_write_ptr(cb_input_sticks);
            const uint32_t page0 = r * tile_h;

            uint64_t bank_addr[bank_period];
            bool affine = false;
            uint64_t stride = 0;
            if constexpr (fast_addrgen == 1) {
                for (uint32_t phase = 0; phase < bank_period; ++phase) {
                    bank_addr[phase] = src.get_noc_addr(page0 + phase, byte_offset);
                }
                // `bank_period < tile_h` is the host-side gate for arming this
                // arm at all, so page `page0 + bank_period` is a stick of this
                // very block and the probe reads nothing extra.
                const uint64_t probe = src.get_noc_addr(page0 + bank_period, byte_offset);
                if (((probe ^ bank_addr[0]) >> 32) == 0 && probe > bank_addr[0]) {
                    stride = probe - bank_addr[0];
                    affine = true;
                }
            }

            uint32_t state_node = 0;
            bool have_state = false;
            uint32_t phase = 0;
            for (uint32_t s = 0; s < tile_h; ++s) {
                uint64_t addr;
                if (affine) {
                    addr = bank_addr[phase];
                    bank_addr[phase] += stride;
                } else {
                    addr = src.get_noc_addr(page0 + s, byte_offset);
                }
                if constexpr (stub_read == 0) {
                    if constexpr (stateful_reads == 1) {
                        const uint32_t node = static_cast<uint32_t>(addr >> 32);
                        if (!have_state || node != state_node) {
                            if constexpr (stateful_one_packet) {
                                noc_async_read_one_packet_set_state(addr, row_bytes);
                            } else {
                                noc_async_read_set_state(addr);
                            }
                            state_node = node;
                            have_state = true;
                        }
                        if constexpr (stateful_one_packet) {
                            noc_async_read_one_packet_with_state(static_cast<uint32_t>(addr), l1_base + s * row_bytes);
                        } else {
                            noc_async_read_with_state(static_cast<uint32_t>(addr), l1_base + s * row_bytes, row_bytes);
                        }
                    } else {
                        noc_async_read(addr, l1_base + s * row_bytes, row_bytes);
                    }
                }
                if (++phase == bank_period) {
                    phase = 0;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_sticks, w);
        } else if constexpr (barrier_per_block == 1 && stub_read == 0 && stagger_reads == 0) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                src,
                /*total_num_rows*/ tile_h,
                /*row_bytes*/ row_bytes,
                /*start_page*/ r * tile_h,
                /*byte_offset_within_page*/ byte_offset);
        } else {
            // Counterfactual / ablation arm, and the `stagger_reads` production arm:
            // identical CB accounting (reserve w, read tile_h sticks at L1 stride
            // row_bytes, push w) — only the ISSUE ORDER of the reads can differ.
            cb_reserve_back(cb_input_sticks, w);
            const uint32_t l1_base = get_write_ptr(cb_input_sticks);
            // R3 read stagger. On a wide-short tensor (`nt_h == 1`) EVERY core reads
            // the same `tile_h` source pages, differing only in byte offset, and the
            // helper issues them in page order — so at any instant the whole fleet
            // is requesting one page, i.e. ONE DRAM bank, and the banks are used
            // one at a time. Rotating the issue order by this core's own block index
            // spreads the in-flight requests across the banks. Byte-for-byte the
            // same transfers to the same L1 destinations: `s` indexes both the
            // source page and its L1 slot, so the block lands stick-ordered no
            // matter which stick is issued first, and the single barrier below still
            // covers all of them.
            const uint32_t s0 = (stagger_reads == 1) ? (b % tile_h) : 0;
            for (uint32_t i = 0; i < tile_h; ++i) {
                uint32_t s = s0 + i;
                if (s >= tile_h) {
                    s -= tile_h;
                }
                if constexpr (stub_read == 0) {
                    noc_async_read(src.get_noc_addr(r * tile_h + s, byte_offset), l1_base + s * row_bytes, row_bytes);
                    if constexpr (barrier_per_block == 0) {
                        noc_async_read_barrier();  // B7 off: one barrier per transaction
                    }
                }
            }
            if constexpr (barrier_per_block == 1) {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, w);
        }
    }
}

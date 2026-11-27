// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu reader (NCRISC).
//
// Cross-core tile-row work stealing (Perf 2; CT steal_mode != STEAL_STATIC, i.e. Plan.work_stealing: some core
// owns more than one block and a channel group has more than one core — otherwise there is nothing to steal:
// every core claims its only block at kernel start):
//
//   Claim protocol (one claim word per core = program semaphore `claim_sem_id`, host-initialized to 0 at
//   every launch, packing F = front claims (bits 0..15) and B = back claims (bits 16..31) on that core's
//   static block range [start_s, start_s + n_s) cut into nb_s = ceil(n_s / block_rows) blocks):
//     owner   : old = fetch_add(own word, 1)        -> block F_old of its own range   iff F_old + B_old < nb
//     stealer : old = fetch_add(victim word, 1<<16) -> block nb - 1 - B_old of victim iff F_old + B_old < nb
//   Race-free: every claim adds 1 to F + B, so the successful claims are exactly the first nb increments
//   in the word's atomic order; front claims take indices 0, 1, ..., back claims nb-1, nb-2, ..., and a
//   front index f and back index nb-1-b can only meet if f + b = nb - 1 — impossible for two claims that
//   both saw F + B < nb (the later one saw the earlier one's increment). Failed claims only inflate the
//   sum (each claimer stops at its first failure on a word), so the 16-bit fields cannot overflow.
//   A core claims a block after it has reserved the block's cb_rows slot (late binding: a claim is made
//   when the core can start loading it). Own blocks are claimed front-first; once the own range is
//   exhausted the core probes every other group member's claim word (one 4 B NoC read
//   each, one barrier) and steals from the BACK of the member with the most unclaimed blocks.
//   Halo: a block contiguous with this core's previous block copies it from the previous slot (L1);
//   any other block with t0 > 0 reads it from DRAM (RM: input rows t0-3..t0-1; TILE: one extra
//   untilized tile-row) — the op's "core-first block, t0 > 0" path; t0 == 0 reads history.
//   Short blocks (a range's ragged last block) are padded to a full slot (push of garbage sticks; compute
//   pops them), so every block occupies exactly one slot of the cb_rows ring whatever order blocks come in.
//   Consumers learn the dynamic schedule from messages:
//     compute: cb_ctrl (CB 8), one page per block = rows | extra << 16, then CTRL_DONE;
//     writer : cb_split_requests, one page per tile-row = {L1 dst, tile-row, rows in block, flags}, then
//              a page with flags = REQ_DONE.
//
// RAW NoC (no dataflow-API wrapper): the claim is noc_fast_atomic_increment<noc_mode, program_ret_addr=true>
// (NOC_AT_INS_INCR_GET with the previous value returned to this core's L1) — the only fetch-and-add on
// the NoC; noc_semaphore_inc issues the same instruction but discards the returned value
// (MEM_NOC_ATOMIC_RET_VAL_ADDR), so it cannot reserve a slot. After the response the command buffer's
// return address is restored to MEM_NOC_ATOMIC_RET_VAL_ADDR for every later noc_semaphore_inc.
// Gap closer: a dataflow-API noc_semaphore_fetch_add(addr, incr) -> previous value.
//
// Block operations owned here (op_design.md -> Block schedule):
//   load_taps        once per channel group (regime tap_multicast_per_group): the group's sender core
//                    (slot 0) reads row 0 of tap p's tile into row 2p of the packed tap tile and multicasts
//                    the packed tiles to the other cores of the group (kernel_lib mcast pipe).
//   load_rows_block  one cb_rows slot per block: sticks [0,3) = halo (tokens t0-3..t0-1),
//                    sticks [3, 3 + 32*rows) = body.
//                    RM:   per tile-row 32 stick reads (+ halo on the first) -> ONE barrier -> push.
//                          Split reader (split_rows > 0): body rows [32 - split_rows, 32) of every
//                          tile-row are read by the writer RISC on the other NoC.
//                    TILE: per tile-row: issue the NEXT input tile-row's reads, then copy the
//                          untilized tile-row (32 contiguous sticks) into the slot -> ONE barrier.
// Raw NoC calls rather than read_sticks_for_tilize (tilize_helpers_dataflow.inl:146-157): that
// helper barriers after every stick and has no notion of a 3-row halo prefix from another source.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (MaybeDeviceZoneScope, PERMANENT): reader_taps_issue, per block reader_rows_reserve,
// reader_halo_issue (+ reader_halo_dram nested when the halo comes from DRAM), per tile-row
// reader_rows_issue, reader_rows_barrier, reader_split_wait; TILE reader_tile_issue,
// reader_untilized_wait, reader_slot_copy. Dynamic mode: reader_claim (own-range claim, per block),
// reader_steal (probe + victim claim, per stolen block and the final failed probe).
#ifdef GQKV_ABLATE_READS
#define GQKV_READ(...)
#else
#define GQKV_READ(...) noc_async_read(__VA_ARGS__)
#endif

namespace {

constexpr uint32_t cb_rows = 0;
constexpr uint32_t cb_tap_tiles = 1;
constexpr uint32_t cb_input_tiles = 4;
constexpr uint32_t cb_untilized_rows = 5;
constexpr uint32_t cb_split_requests = 6;  // reader -> writer, one page per tile-row
constexpr uint32_t cb_split_done = 7;      // writer -> reader, one token per tile-row (split_rows > 0)
constexpr uint32_t cb_ctrl = 8;            // dynamic: reader -> compute, one page per block
constexpr uint32_t cb_steal_scratch = 9;   // dynamic: atomic return word (page 0) + probe words (page 1 + v)

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t TAP_COUNT = 4;
constexpr uint32_t HALO_ROWS = TAP_COUNT - 1;
constexpr uint32_t TAP_ROW_STRIDE = 2;

constexpr uint32_t STEAL_STATIC = 0;  // nothing to steal: the static ranges, no protocol
constexpr uint32_t STEAL_ON = 1;      // own claims, then steal from the back of the group's other ranges
constexpr uint32_t STEAL_EAGER = 2;   // test hook only: steal first, own range after (maximal claim contention)

constexpr uint32_t CTRL_DONE = 0xFFFFFFFFu;
constexpr uint32_t REQ_DONE = 1u;
constexpr uint32_t CLAIM_SENTINEL = 0xFFFFFFFFu;
constexpr uint32_t FRONT_UNIT = 1u;
constexpr uint32_t BACK_UNIT = 1u << 16;
constexpr uint32_t NO_ROW = 0xFFFFFFFFu;
constexpr uint32_t SCRATCH_STRIDE = 16;  // one L1-aligned slot per word (NoC read alignment)

FORCE_INLINE void copy_local(const Noc& noc, uint32_t src_addr, uint32_t dst_addr, uint32_t size_bytes) {
    UnicastEndpoint self_ep;
    noc.async_read(
        self_ep,
        CoreLocalMem<uint32_t>(dst_addr),
        size_bytes,
        dataflow_kernel_lib::local_addr(src_addr, noc.get_noc_id()),
        {});
}

template <uint32_t block_ch_tiles, uint32_t tile_bytes, uint32_t elem_bytes, typename Accessor>
FORCE_INLINE void issue_tap_rows(
    const Accessor& tap_acc, uint32_t tap_idx, uint32_t tap_base, uint32_t ch_group_tile_start) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    const uint32_t row_offset = TAP_ROW_STRIDE * tap_idx * face_row_bytes;
    for (uint32_t c = 0; c < block_ch_tiles; ++c) {
        const uint64_t src = tap_acc.get_noc_addr(ch_group_tile_start + c);
        const uint32_t dst = tap_base + c * tile_bytes + row_offset;
        noc_async_read(src, dst, face_row_bytes);
        noc_async_read(src + face_bytes, dst + face_bytes, face_row_bytes);
    }
}

template <uint32_t block_ch_tiles, uint32_t history_page_channels, uint32_t elem_bytes, typename Accessor>
FORCE_INLINE void read_history_halo(const Accessor& history_acc, uint32_t slot_base, uint32_t ch_group_tile_start) {
    constexpr uint32_t group_channels = block_ch_tiles * TILE_DIM;
    constexpr uint32_t stick_bytes = group_channels * elem_bytes;
    const uint32_t c0 = ch_group_tile_start * TILE_DIM;
    for (uint32_t r = 0; r < HALO_ROWS; ++r) {
        const uint32_t dst_row = slot_base + r * stick_bytes;
        if constexpr (history_page_channels == 0) {
            GQKV_READ(history_acc.get_noc_addr(r, c0 * elem_bytes), dst_row, stick_bytes);
        } else {
            constexpr uint32_t P = history_page_channels;
            for (uint32_t c = c0; c < c0 + group_channels;) {
                const uint32_t page = c / P;
                const uint32_t in_page = c - page * P;
                const uint32_t run_end = (page + 1) * P < c0 + group_channels ? (page + 1) * P : c0 + group_channels;
                GQKV_READ(
                    history_acc.get_noc_addr(page, (r * P + in_page) * elem_bytes),
                    dst_row + (c - c0) * elem_bytes,
                    (run_end - c) * elem_bytes);
                c = run_end;
            }
        }
    }
}

// Atomic fetch-and-add on a (possibly remote, possibly own) claim word; returns the previous value.
FORCE_INLINE uint32_t claim_fetch_add(uint64_t word_noc_addr, uint32_t incr, uint32_t ret_addr) {
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr);
    *ret = CLAIM_SENTINEL;
    asm volatile("fence" ::: "memory");  // the sentinel is in L1 before the response can land
    noc_fast_atomic_increment<noc_mode, /*program_ret_addr=*/true>(
        noc_index,
        write_at_cmd_buf,
        word_noc_addr,
        NOC_UNICAST_WRITE_VC,
        incr,
        31 /*wrap: none*/,
        false /*linked*/,
        false /*posted*/,
        ret_addr);
    noc_async_atomic_barrier();
    // Later noc_semaphore_inc calls (tap multicast) return into the default scratch word again.
    NOC_CMD_BUF_WRITE_REG(noc_index, write_at_cmd_buf, NOC_RET_ADDR_LO, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    uint32_t v;
    do {
        invalidate_l1_cache();
        v = *ret;
    } while (v == CLAIM_SENTINEL);
    return v;
}

}  // namespace

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr bool tile_input = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t history_page_channels = get_compile_time_arg_val(6);
    constexpr uint32_t split_rows = get_compile_time_arg_val(7);
    constexpr bool stagger_row_walk = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t steal_mode = get_compile_time_arg_val(9);
    constexpr uint32_t steal_min = get_compile_time_arg_val(10);  // min unclaimed blocks to steal from a range
    constexpr uint32_t claim_sem_id = get_compile_time_arg_val(11);
    constexpr auto input_args = TensorAccessorArgs<12>();
    constexpr auto history_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto taps_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("taps_mcast_ct_offset"),
        get_named_compile_time_arg_val("taps_mcast_rt_offset")>();
    constexpr bool dynamic = steal_mode != STEAL_STATIC;

    constexpr uint32_t channel_tile_bytes = TILE_DIM * elem_bytes;
    constexpr uint32_t stick_bytes = block_ch_tiles * channel_tile_bytes;
    constexpr uint32_t untilized_row_bytes = TILE_DIM * stick_bytes;
    constexpr uint32_t first_row_push = TILE_DIM + HALO_ROWS;
    constexpr uint32_t tap_tiles_bytes = block_ch_tiles * tile_bytes;
    constexpr uint32_t own_rows = TILE_DIM - split_rows;
    constexpr uint32_t full_slot_sticks = TILE_DIM * block_rows + HALO_ROWS;
    static_assert(split_rows <= TILE_DIM, "split_rows is a row count inside one tile-row");
    static_assert(split_rows == 0 || !tile_input, "the split reader is a ROW_MAJOR-regime lever");

    // ---- runtime args ----
    uint32_t arg = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t history_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t input_col_offset_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);
    // dynamic-mode group geometry (present for every mode; unused when static)
    const uint32_t group_first_core = get_arg_val<uint32_t>(arg++);  // logical row-major index of slot 0
    const uint32_t cores_per_group = get_arg_val<uint32_t>(arg++);
    const uint32_t own_slot = get_arg_val<uint32_t>(arg++);
    const uint32_t range_base = get_arg_val<uint32_t>(arg++);  // slot s: base + (s < rem) tile-rows
    const uint32_t range_rem = get_arg_val<uint32_t>(arg++);
    const uint32_t grid_x = get_arg_val<uint32_t>(arg++);
    const uint32_t grid_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t x_table = arg;  // virtual NoC x of each logical column, then y of each logical row
    const uint32_t y_table = arg + grid_x;
    (void)grid_rows;

    const uint32_t num_blocks_this_core = (core_seq_tiles + block_rows - 1) / block_rows;
    const uint32_t last_block_rows = core_seq_tiles - (num_blocks_this_core - 1) * block_rows;

    const auto input_acc = TensorAccessor(input_args, input_addr);
    const auto history_acc = TensorAccessor(history_args, history_addr);
    Noc noc;

    const uint32_t input_byte_offset = (input_col_offset_tiles + ch_group_tile_start) * channel_tile_bytes;

    // ---- claim protocol state (dynamic) ----
    const uint32_t claim_word_addr = get_semaphore(claim_sem_id);
    const uint32_t scratch_base = dynamic ? get_write_ptr(cb_steal_scratch) : 0;
    auto slot_range = [&](uint32_t s, uint32_t& start, uint32_t& n) {
        start = s * range_base + (s < range_rem ? s : range_rem);
        n = range_base + (s < range_rem ? 1 : 0);
    };
    auto slot_blocks = [&](uint32_t s) {
        uint32_t start, n;
        slot_range(s, start, n);
        return (n + block_rows - 1) / block_rows;
    };
    auto slot_word = [&](uint32_t s) -> uint64_t {
        const uint32_t i = group_first_core + s;
        const uint32_t x = get_arg_val<uint32_t>(x_table + i % grid_x);
        const uint32_t y = get_arg_val<uint32_t>(y_table + i / grid_x);
        return get_noc_addr(x, y, claim_word_addr, noc_index);
    };
    auto block_of = [&](uint32_t s, uint32_t k, uint32_t& first_row, uint32_t& rows) {
        uint32_t start, n;
        slot_range(s, start, n);
        first_row = start + k * block_rows;
        const uint32_t left = n - k * block_rows;
        rows = left < block_rows ? left : block_rows;
    };
    bool own_live = true;
    uint32_t stolen_blocks = 0;
    // Own range, front first.
    auto claim_own = [&](uint32_t& first_row, uint32_t& rows) -> bool {
        if (!own_live) {
            return false;
        }
        MaybeDeviceZoneScope("reader_claim");
        const uint32_t nb = slot_blocks(own_slot);
        const uint32_t old = claim_fetch_add(slot_word(own_slot), FRONT_UNIT, scratch_base);
        const uint32_t f = old & 0xFFFFu, b = old >> 16;
        if (f + b < nb) {
            block_of(own_slot, f, first_row, rows);
            return true;
        }
        own_live = false;
        return false;
    };
    // Back of the group member with the most unclaimed blocks (probe = one 4 B read per member).
    auto claim_stolen = [&](uint32_t& first_row, uint32_t& rows) -> bool {
        MaybeDeviceZoneScope("reader_steal");
        for (;;) {
            for (uint32_t v = 0; v < cores_per_group; ++v) {
                if (v != own_slot) {
                    noc_async_read(slot_word(v), scratch_base + SCRATCH_STRIDE * (v + 1), sizeof(uint32_t));
                }
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            uint32_t best = cores_per_group, best_left = 0;
            for (uint32_t v = 0; v < cores_per_group; ++v) {
                if (v == own_slot) {
                    continue;
                }
                const uint32_t w =
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base + SCRATCH_STRIDE * (v + 1));
                const uint32_t used = (w & 0xFFFFu) + (w >> 16);
                const uint32_t nb = slot_blocks(v);
                const uint32_t left = used < nb ? nb - used : 0;
                if (left >= steal_min && left > best_left) {
                    best = v;
                    best_left = left;
                }
            }
            if (best == cores_per_group) {
                return false;  // F + B only grows: nothing the probe saw exhausted can come back
            }
            const uint32_t nb = slot_blocks(best);
            const uint32_t old = claim_fetch_add(slot_word(best), BACK_UNIT, scratch_base);
            const uint32_t f = old & 0xFFFFu, b = old >> 16;
            if (f + b < nb) {
                block_of(best, nb - 1 - b, first_row, rows);
                ++stolen_blocks;
                return true;
            }
        }
    };
    // Claim the next block for this core; false when the group has nothing left for it.
    auto claim_next = [&](uint32_t& first_row, uint32_t& rows) -> bool {
        if constexpr (steal_mode == STEAL_EAGER) {
            return claim_stolen(first_row, rows) || claim_own(first_row, rows);
        }
        return claim_own(first_row, rows) || claim_stolen(first_row, rows);
    };

    // ---- load_taps ----
    auto tap_sender = taps_mcast.optional_sender(noc);
    auto tap_receiver = taps_mcast.optional_receiver(noc);
    cb_reserve_back(cb_tap_tiles, block_ch_tiles);
    const uint32_t tap_base = get_write_ptr(cb_tap_tiles);
    if (tap_sender) {
        MaybeDeviceZoneScope("reader_taps_issue");
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap0_args, tap0_addr), 0, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap1_args, tap1_addr), 1, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap2_args, tap2_addr), 2, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap3_args, tap3_addr), 3, tap_base, ch_group_tile_start);
    }
    bool taps_pending = true;
    auto publish_taps_and_barrier = [&]() {
        if (tap_sender) {
            noc_async_read_barrier();
            tap_sender->send(tap_base, tap_base, tap_tiles_bytes);
        } else if (tap_receiver) {
            tap_receiver->receive();
        }
        noc_async_read_barrier();
        cb_push_back(cb_tap_tiles, block_ch_tiles);
        taps_pending = false;
    };

    auto post_request = [&](uint32_t dst, uint32_t row, uint32_t rows, uint32_t flags) {
        cb_reserve_back(cb_split_requests, 1);
        volatile tt_l1_ptr uint32_t* page =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_split_requests));
        page[0] = dst;
        if constexpr (dynamic) {
            page[1] = row;
            page[2] = rows;
            page[3] = flags;
        }
        cb_push_back(cb_split_requests, 1);
    };
    auto post_ctrl = [&](uint32_t value) {
        cb_reserve_back(cb_ctrl, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl)) = value;
        cb_push_back(cb_ctrl, 1);
    };

    uint32_t prev_slot_base = 0;
    uint32_t prev_rows = 0;
    uint32_t prev_end_row = NO_ROW;  // first tile-row after this core's previous block
    for (uint32_t block_idx = 0;; ++block_idx) {
        uint32_t block_seq_start, rows_this_block;
        if constexpr (!dynamic) {
            if (block_idx == num_blocks_this_core) {
                break;
            }
            rows_this_block = (block_idx + 1 == num_blocks_this_core) ? last_block_rows : block_rows;
            block_seq_start = core_seq_start + block_idx * block_rows;
            MaybeDeviceZoneScope("reader_rows_reserve");
            cb_reserve_back(cb_rows, TILE_DIM * rows_this_block + HALO_ROWS);
        } else {
            {
                MaybeDeviceZoneScope("reader_rows_reserve");
                cb_reserve_back(cb_rows, full_slot_sticks);
            }
            if (!claim_next(block_seq_start, rows_this_block)) {
                break;
            }
        }
        const uint32_t slot_base = get_write_ptr(cb_rows);
        const bool contiguous = block_idx > 0 && block_seq_start == prev_end_row;

        bool halo_from_untilized = false;
        if constexpr (dynamic) {
            const uint32_t extra = (tile_input && block_seq_start > 0 && !contiguous) ? 1 : 0;
            post_ctrl(rows_this_block | (extra << 16));
            if constexpr (split_rows == 0) {
                for (uint32_t m = 0; m < rows_this_block; ++m) {
                    post_request(0, block_seq_start + m, rows_this_block, 0);
                }
            }
        }
        {
            MaybeDeviceZoneScope("reader_halo_issue");
            if (block_seq_start == 0) {
                read_history_halo<block_ch_tiles, history_page_channels, elem_bytes>(
                    history_acc, slot_base, ch_group_tile_start);
            } else if (contiguous) {
                // The previous block's last 3 body sticks sit at [32 * prev_rows, +3) of its slot.
                copy_local(
                    noc, prev_slot_base + TILE_DIM * prev_rows * stick_bytes, slot_base, HALO_ROWS * stick_bytes);
            } else if constexpr (!tile_input) {
                MaybeDeviceZoneScope("reader_halo_dram");
                const uint32_t first_token = block_seq_start * TILE_DIM - HALO_ROWS;
                for (uint32_t r = 0; r < HALO_ROWS; ++r) {
                    GQKV_READ(
                        input_acc.get_noc_addr(first_token + r, input_byte_offset),
                        slot_base + r * stick_bytes,
                        stick_bytes);
                }
            } else {
                halo_from_untilized = true;
            }
        }

        if constexpr (!tile_input) {
            for (uint32_t m = 0; m < rows_this_block; ++m) {
                const uint32_t token0 = (block_seq_start + m) * TILE_DIM;
                const uint32_t dst = slot_base + (HALO_ROWS + TILE_DIM * m) * stick_bytes;
                if constexpr (split_rows > 0) {
                    post_request(dst, block_seq_start + m, rows_this_block, 0);
                }
                {
                    MaybeDeviceZoneScope("reader_rows_issue");
                    for (uint32_t j = 0; j < own_rows; ++j) {
                        const uint32_t i = stagger_row_walk ? (j + ch_group_tile_start + core_seq_start) % own_rows : j;
                        GQKV_READ(
                            input_acc.get_noc_addr(token0 + i, input_byte_offset), dst + i * stick_bytes, stick_bytes);
                    }
                }
                {
                    MaybeDeviceZoneScope("reader_rows_barrier");
                    if (taps_pending) {
                        publish_taps_and_barrier();
                    } else {
                        noc_async_read_barrier();
                    }
                }
                if constexpr (split_rows > 0) {
                    MaybeDeviceZoneScope("reader_split_wait");
                    cb_wait_front(cb_split_done, 1);
                    cb_pop_front(cb_split_done, 1);
                }
                cb_push_back(cb_rows, m == 0 ? first_row_push : TILE_DIM);
            }
        } else {
            const uint32_t extra_rows = halo_from_untilized ? 1 : 0;
            const uint32_t rows_to_untilize = rows_this_block + extra_rows;
            const uint32_t first_tile_row = block_seq_start - extra_rows;
            auto issue_input_tile_row = [&](uint32_t tile_row) {
                MaybeDeviceZoneScope("reader_tile_issue");
                cb_reserve_back(cb_input_tiles, block_ch_tiles);
                const uint32_t dst = get_write_ptr(cb_input_tiles);
                const uint32_t first_page = tile_row * input_width_tiles + input_col_offset_tiles + ch_group_tile_start;
                for (uint32_t c = 0; c < block_ch_tiles; ++c) {
                    GQKV_READ(input_acc.get_noc_addr(first_page + c), dst + c * tile_bytes, tile_bytes);
                }
            };

            issue_input_tile_row(first_tile_row);
            if (taps_pending) {
                publish_taps_and_barrier();
            } else {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_tiles, block_ch_tiles);
            for (uint32_t j = 0; j < rows_to_untilize; ++j) {
                const bool next_pending = (j + 1 < rows_to_untilize);
                if (next_pending) {
                    issue_input_tile_row(first_tile_row + j + 1);
                }
                {
                    MaybeDeviceZoneScope("reader_untilized_wait");
                    cb_wait_front(cb_untilized_rows, block_ch_tiles);
                }
                MaybeDeviceZoneScope("reader_slot_copy");
                const uint32_t src = get_read_ptr(cb_untilized_rows);
                if (j < extra_rows) {
                    copy_local(noc, src + (TILE_DIM - HALO_ROWS) * stick_bytes, slot_base, HALO_ROWS * stick_bytes);
                } else {
                    const uint32_t m = j - extra_rows;
                    copy_local(noc, src, slot_base + (HALO_ROWS + TILE_DIM * m) * stick_bytes, untilized_row_bytes);
                }
                noc_async_read_barrier();
                if (next_pending) {
                    cb_push_back(cb_input_tiles, block_ch_tiles);
                }
                cb_pop_front(cb_untilized_rows, block_ch_tiles);
                if (j >= extra_rows) {
                    cb_push_back(cb_rows, j == extra_rows ? first_row_push : TILE_DIM);
                }
            }
        }
        if constexpr (dynamic) {
            if (rows_this_block < block_rows) {
                // ragged block: pad to a full slot so the ring stays slot-aligned (compute pops the pad)
                cb_push_back(cb_rows, TILE_DIM * (block_rows - rows_this_block));
            }
        }
        prev_slot_base = slot_base;
        prev_rows = rows_this_block;
        prev_end_row = block_seq_start + rows_this_block;
    }
    if constexpr (dynamic) {
        if (taps_pending) {
            publish_taps_and_barrier();  // a core left without blocks still serves / receives the taps
        }
        post_ctrl(CTRL_DONE);
        post_request(0, 0, 0, REQ_DONE);
        DeviceTimestampedData("steal_blocks", stolen_blocks);
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}

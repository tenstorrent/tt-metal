// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Halo multicast (the descriptor's HALO_MCAST), sender half: an owner core pushes its input shard's rows to the cores
// of its column whose slabs read them, into their input rings at slot t % HALO_RING_ROWS (input tile-row t sits at the
// same L1 address on every core), then closes with its flag (semaphore HALO_SEM0 + o % HALO_FLAGS). The reader sends
// down the column on NoC0, the writer up on NoC1; both call send() with the runs of their direction.
//
// Runtime args from `a`: n_runs, then per run (row_lo, row_hi, dests, sx, sy, ex, ey), then the flag's rectangle
// (dests, sx, sy, ex, ey). A rectangle is a column segment in the NoC's multicast order (NoC0 low corner first, NoC1
// high corner first); the flag's covers every run's.
//
// RAW multicast (capability): dataflow_kernel_lib's SenderPipe / ReceiverPipe model one fixed sender per channel with a
// receiver ack before every send. Here every owner of a column sends (8 senders x 2 directions per column), each into
// its own column segment, with no ack: the ring slots are free at kernel start (one M-block per core), and the flags
// are dispatch-initialized semaphores. A channel per (owner, direction) would take ~16 semaphores per core, and the
// helper has no non-blocking send (one packet when the command buffer is free) for the interleaved form.
//
// Measured (Blackhole p150b, aspp.convs.1 @ LoFi, bit-identical in every mode): a REGRESSION against the pull fill in
// every form. Zoned, one run each vs a same-session baseline of 47.9-48.6 us: linked shard chains 55.3, unlinked 63.4,
// no path reserve 53.4, unicast push 53.6; exact runs 57.1 / 56.0 / 51.3 / 51.8; interleaved 61.6 / 50.8 (shard), 51.7
// / 50.0 (exact); progress flags by atomic increment 59.8-60.6, by value writes 49.2-49.8, top-down 49.4-49.9.
// Unzoned medians of 3: 47786 -> 48950 ns (best: no path reserve, exact, interleaved, 4-row value flags). The pushes
// halve the column's NoC0 link bytes and end the fill no later, but a row's readers use it as different kernel rows
// (needed at different times), so every reader's first band lands later than its ordered pull delivered it and the
// tap-shift prelude ends 1-2 us later.
//
// HALO_MODE (bench switch): 0 linked chains with path reserve (the API default for a chain), 1 unlinked packets with
// path reserve, 2 unlinked packets without path reserve, 3 unicast writes to each core of the segment.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef HALO_MODE
#define HALO_MODE 0
#endif
// Rows per progress flag of the reader's interleaved down multicast (0: one flag once the shard is in). With progress
// flags the flag counts the shard's complete rows (an atomic increment after every piece of at most this many rows).
#ifndef HALO_FLAG_ROWS
#define HALO_FLAG_ROWS 0
#endif
// The interleaved down multicast sends its rows top-down (the rows the farthest readers take as their first band
// first); with progress flags the flag then counts the shard's complete rows from its top.
#ifndef HALO_DESC
#define HALO_DESC 0
#endif

namespace conv2d_3x3::halo {

template <uint32_t S>
FORCE_INLINE uint32_t slot(uint32_t t) {
    return t % S;
}

// One multicast (or, in mode 3, the unicast writes of every core of the segment).
FORCE_INLINE void put(
    uint32_t src,
    uint32_t dst,
    uint32_t bytes,
    uint32_t dests,
    uint32_t sx,
    uint32_t sy,
    uint32_t ex,
    uint32_t ey,
    bool linked) {
#if HALO_MODE == 3
    const int32_t step = ey >= sy ? 1 : -1;
    for (int32_t y = static_cast<int32_t>(sy);; y += step) {
        noc_async_write(src, get_noc_addr(sx, static_cast<uint32_t>(y), dst), bytes);
        if (y == static_cast<int32_t>(ey)) {
            break;
        }
    }
#elif HALO_MODE == 0
    noc_async_write_multicast(src, get_noc_multicast_addr(sx, sy, ex, ey, dst), bytes, dests, linked);
#else
    ncrisc_noc_fast_write_any_len<noc_mode>(
        noc_index,
        write_cmd_buf,
        src,
        get_noc_multicast_addr(sx, sy, ex, ey, dst),
        bytes,
        NOC_MULTICAST_WRITE_VC,
        true,
        false,
        dests,
        HALO_MODE == 1);
#endif
}

// Push this owner's shard rows (input tile-rows [own_lo, own_lo + R) at shard_l1, row_bytes apart) along the runs at
// runtime arg `a`. move_bytes < row_bytes: a short channel slice, one row per write.
template <uint32_t S>
FORCE_INLINE void send(
    uint32_t a,
    uint32_t shard_l1,
    uint32_t own_lo,
    uint32_t ring_base,
    uint32_t row_bytes,
    uint32_t move_bytes,
    uint32_t flag,
    uint32_t shard_len) {
    const uint32_t n_runs = get_arg_val<uint32_t>(a);
    if (n_runs == 0) {
        return;
    }
    // The flag counts the shard's complete rows (HALO_FLAG_ROWS) or is 1 once all are in.
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) = HALO_FLAG_ROWS ? shard_len : 1;
    const uint32_t fa = a + 1 + 7 * n_runs;
    const uint32_t f_dests = get_arg_val<uint32_t>(fa);
    const uint32_t fsx = get_arg_val<uint32_t>(fa + 1), fsy = get_arg_val<uint32_t>(fa + 2);
    const uint32_t fex = get_arg_val<uint32_t>(fa + 3), fey = get_arg_val<uint32_t>(fa + 4);
    const bool per_row = move_bytes < row_bytes;
    a += 1;
    for (uint32_t i = 0; i < n_runs; ++i, a += 7) {
        const uint32_t r_lo = get_arg_val<uint32_t>(a);
        const uint32_t r_hi = get_arg_val<uint32_t>(a + 1);
        const uint32_t dests = get_arg_val<uint32_t>(a + 2);
        const uint32_t sx = get_arg_val<uint32_t>(a + 3), sy = get_arg_val<uint32_t>(a + 4);
        const uint32_t ex = get_arg_val<uint32_t>(a + 5), ey = get_arg_val<uint32_t>(a + 6);
        // A linked chain stays open into the flag where the flag goes to the same rectangle.
        const bool close_by_flag = i + 1 == n_runs && dests == f_dests && ex == fex && ey == fey;
        for (uint32_t t = r_lo; t < r_hi;) {
            uint32_t n = per_row ? 1 : r_hi - t;
            const uint32_t to_end = S - slot<S>(t);
            n = n < to_end ? n : to_end;
            put(shard_l1 + (t - own_lo) * row_bytes,
                ring_base + slot<S>(t) * row_bytes,
                per_row ? move_bytes : n * row_bytes,
                dests,
                sx,
                sy,
                ex,
                ey,
                t + n < r_hi || close_by_flag);
            t += n;
        }
    }
#if HALO_MODE == 3
    // Unicast: the flag to each core of the segment, after its data (same NoC and VC).
    noc_async_writes_flushed();
    const int32_t step = fey >= fsy ? 1 : -1;
    for (int32_t y = static_cast<int32_t>(fsy);; y += step) {
        noc_async_write_barrier();
        noc_semaphore_set_remote(flag, get_noc_addr(fsx, static_cast<uint32_t>(y), flag));
        if (y == static_cast<int32_t>(fey)) {
            break;
        }
    }
#else
    noc_semaphore_set_multicast(flag, get_noc_multicast_addr(fsx, fsy, fex, fey, flag), f_dests, false);
#endif
}

// Non-blocking form of send() (multicast modes 0..2): step() issues at most one packet, and only when the write command
// buffer is free, so the reader can push the rows that already landed between packets (HALO_INTERLEAVE).
template <uint32_t S>
struct Sender {
    uint32_t a = 0, fa = 0, n_runs = 0, run = 0;
    uint32_t t = 0, r_hi = 0, dests = 0, sx = 0, sy = 0, ex = 0, ey = 0;
    bool close_by_flag = false, done = true;
    // The current contiguous piece: rows [t, t + n) (one row for a short slice), bytes_left of it unsent.
    uint32_t src = 0, dst = 0, bytes_left = 0, piece_rows = 0;
    uint32_t shard_l1 = 0, own_lo = 0, ring_base = 0, row_bytes = 0, move_bytes = 0, flag = 0;
    uint32_t shard_len = 0, reported = 0;  // progress flags: rows of the shard reported complete
    uint64_t flag_mcast = 0;
    uint32_t flag_dests = 0;
    uint32_t values_l1 = 0;  // HALO_CB_VALUES: word i holds i (the absolute-value progress flags' sources)

    uint32_t r_lo = 0, runs_left = 0;
    FORCE_INLINE void load_run() {
        const uint32_t b = a + 1 + 7 * run;
        r_lo = get_arg_val<uint32_t>(b);
        r_hi = get_arg_val<uint32_t>(b + 1);
        t = HALO_DESC ? r_hi : r_lo;
        dests = get_arg_val<uint32_t>(b + 2);
        sx = get_arg_val<uint32_t>(b + 3);
        sy = get_arg_val<uint32_t>(b + 4);
        ex = get_arg_val<uint32_t>(b + 5);
        ey = get_arg_val<uint32_t>(b + 6);
        close_by_flag = (HALO_DESC ? run == 0 : run + 1 == n_runs) && dests == get_arg_val<uint32_t>(fa) &&
                        ex == get_arg_val<uint32_t>(fa + 3) && ey == get_arg_val<uint32_t>(fa + 4);
        load_piece();
    }
    FORCE_INLINE void load_piece() {
#if HALO_DESC
        // Rows [t - n, t): consecutive slots (slot(t - 1) - n + 1 >= 0), at most HALO_FLAG_ROWS of them.
        uint32_t n = move_bytes < row_bytes ? 1 : t - r_lo;
        const uint32_t to_start = slot<S>(t - 1) + 1;
        n = n < to_start ? n : to_start;
        if constexpr (HALO_FLAG_ROWS > 0) {
            n = n < HALO_FLAG_ROWS ? n : HALO_FLAG_ROWS;
        }
        piece_rows = n;
        src = shard_l1 + (t - n - own_lo) * row_bytes;
        dst = ring_base + slot<S>(t - n) * row_bytes;
        bytes_left = move_bytes < row_bytes ? move_bytes : n * row_bytes;
#else
        uint32_t n = move_bytes < row_bytes ? 1 : r_hi - t;
        const uint32_t to_end = S - slot<S>(t);
        n = n < to_end ? n : to_end;
        if constexpr (HALO_FLAG_ROWS > 0) {
            n = n < HALO_FLAG_ROWS ? n : HALO_FLAG_ROWS;
        }
        piece_rows = n;
        src = shard_l1 + (t - own_lo) * row_bytes;
        dst = ring_base + slot<S>(t) * row_bytes;
        bytes_left = move_bytes < row_bytes ? move_bytes : n * row_bytes;
#endif
    }
    FORCE_INLINE void init(
        uint32_t arg,
        uint32_t shard_l1_,
        uint32_t own_lo_,
        uint32_t ring_base_,
        uint32_t row_bytes_,
        uint32_t move_bytes_,
        uint32_t flag_,
        uint32_t shard_len_) {
        a = arg;
        n_runs = get_arg_val<uint32_t>(a);
        done = n_runs == 0;
        if (done) {
            return;
        }
        shard_l1 = shard_l1_, own_lo = own_lo_, ring_base = ring_base_, row_bytes = row_bytes_;
        move_bytes = move_bytes_, flag = flag_, shard_len = shard_len_, reported = 0;
#ifdef HALO_CB_VALUES
        values_l1 = get_write_ptr(HALO_CB_VALUES);
        for (uint32_t i = 0; i <= shard_len; ++i) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(values_l1)[i] = i;
        }
#endif
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) = HALO_FLAG_ROWS ? shard_len : 1;
        fa = a + 1 + 7 * n_runs;
        runs_left = n_runs;
        flag_dests = get_arg_val<uint32_t>(fa);
        flag_mcast = get_noc_multicast_addr(
            get_arg_val<uint32_t>(fa + 1),
            get_arg_val<uint32_t>(fa + 2),
            get_arg_val<uint32_t>(fa + 3),
            get_arg_val<uint32_t>(fa + 4),
            flag);
        run = HALO_DESC ? n_runs - 1 : 0;
        load_run();
    }
    // Issue the next packet if the command buffer is free; true once everything (the flag too) is issued.
    FORCE_INLINE bool step() {
        if (done) {
            return true;
        }
        if (runs_left == 0) {
            if constexpr (HALO_FLAG_ROWS > 0) {
                if (reported < shard_len) {
#ifdef HALO_CB_VALUES
                    noc_semaphore_set_multicast(values_l1 + 4 * shard_len, flag_mcast, flag_dests, false);
#else
                    noc_semaphore_inc_multicast(flag_mcast, shard_len - reported, flag_dests);
#endif
                }
            } else {
                noc_semaphore_set_multicast(flag, flag_mcast, flag_dests, false);
            }
            done = true;
            return true;
        }
        if (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            return false;
        }
        const uint32_t len = bytes_left < NOC_MAX_BURST_SIZE ? bytes_left : NOC_MAX_BURST_SIZE;
        const bool last_of_run = len == bytes_left && (HALO_DESC ? t - piece_rows <= r_lo : t + piece_rows >= r_hi);
        ncrisc_noc_fast_write<noc_mode>(
            noc_index,
            write_cmd_buf,
            src,
            get_noc_multicast_addr(sx, sy, ex, ey, dst),
            len,
            NOC_MULTICAST_WRITE_VC,
            true,
            HALO_MODE == 0 && (!last_of_run || close_by_flag),
            dests,
            HALO_MODE != 2);
        src += len;
        dst += len;
        bytes_left -= len;
        if (bytes_left == 0) {
            t = HALO_DESC ? t - piece_rows : t + piece_rows;
            if constexpr (HALO_FLAG_ROWS > 0) {
                // Rows below t (top-down: from t up) are complete (rows no down receiver reads count as complete).
                const uint32_t upto = HALO_DESC ? own_lo + shard_len - t : t - own_lo;
                if (upto > reported) {
#ifdef HALO_CB_VALUES
                    // An absolute-value flag write (word `upto` of the values table), as the data: same NoC and VC.
                    while (!noc_cmd_buf_ready(noc_index, write_reg_cmd_buf)) {
                    }
                    ncrisc_noc_fast_write<noc_mode>(
                        noc_index,
                        write_reg_cmd_buf,
                        values_l1 + 4 * upto,
                        flag_mcast,
                        4,
                        NOC_MULTICAST_WRITE_VC,
                        true,
                        false,
                        flag_dests,
                        HALO_MODE != 2);
#else
                    while (!noc_cmd_buf_ready(noc_index, write_at_cmd_buf)) {
                    }
                    noc_semaphore_inc_multicast(flag_mcast, upto - reported, flag_dests);
#endif
                    reported = upto;
                }
            }
            if (HALO_DESC ? t > r_lo : t < r_hi) {
                load_piece();
            } else if (--runs_left > 0) {
                run = HALO_DESC ? run - 1 : run + 1;
                load_run();
            }
        }
        return false;
    }
};

}  // namespace conv2d_3x3::halo

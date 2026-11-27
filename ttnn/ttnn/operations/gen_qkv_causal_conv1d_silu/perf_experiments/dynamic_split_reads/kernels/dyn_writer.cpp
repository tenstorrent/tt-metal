// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu writer (BRISC / NoC1) — perf experiment `dynamic_split_reads`.
//
// store_block exactly as in the op's writer: per output group wait, issue the q / k / v writes,
// flush, pop. The ONE change is the ROW_MAJOR split reader (split_rows > 0 turns it on): instead
// of serving a static share of every tile-row before each output wait, this kernel claims body
// sticks of the reader's current tile-row FROM THE BACK, and only while it is idle — whenever the
// output group it waits for is not there yet, it polls the reader's shared word R (see
// dyn_reader.cpp for the protocol). Claims are gated by NoC1 completions (at most writer_window
// of this RISC's reads outstanding), so a core whose NoC1 is congested reads fewer sticks and the
// reader (NoC0) takes the rest.
//
// poll() handles at most one tile-row: ENTER -> wait for ACK / CLOSED -> claims from the back
// until it meets the reader, the reader leaves the row, or its own output becomes ready -> DONE. It does NOT wait for
// its reads to land: with DONE it publishes N = this RISC's NoC1 read-issue count after its last claim, and the reader
// waits until this core's NoC1 read-response counter reaches N (no other RISC reads on NoC1 here, and this kernel
// issues no further reads until the reader opens the next tile-row, which it does only after that wait). So the writer
// goes straight back to draining output.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (MaybeDeviceZoneScope): per output group writer_out_wait (includes any split claims
// served while waiting), writer_out_issue, writer_out_flush; per served tile-row writer_split_ack
// (ENTER -> ACK / CLOSED), writer_split_issue (claims).
#ifdef GQKV_ABLATE_READS
#define GQKV_READ(...)
#else
#define GQKV_READ(...) noc_async_read(__VA_ARGS__)
#endif
#ifdef GQKV_ABLATE_WRITES
#define GQKV_WRITE(...)
#else
#define GQKV_WRITE(...) noc_async_write(__VA_ARGS__)
#endif

namespace {

constexpr uint32_t cb_output_tiles = 3;
constexpr uint32_t cb_split_requests = 6;  // reader-owned shared words R (+0) and D (+16)
constexpr uint32_t cb_split_done = 7;      // this kernel's shared words W (+0), N (+4)

constexpr uint32_t TILE_DIM = 32;

constexpr uint32_t R_OPEN = 0;
constexpr uint32_t R_ACK = 1;
constexpr uint32_t R_CLOSED = 2;
constexpr uint32_t W_ENTER = 1;
constexpr uint32_t W_CLAIM = 2;
constexpr uint32_t W_DONE = 3;

FORCE_INLINE uint32_t reads_outstanding() {
    return noc_reads_num_issued[noc_index] - NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
}

// This RISC's output writes not yet acked on its NoC.
FORCE_INLINE uint32_t writes_unacked() {
    return noc_nonposted_writes_acked[noc_index] - NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    [[maybe_unused]] constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t split_rows = get_compile_time_arg_val(6);       // > 0: the dynamic split is on
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);      // one token row of this core's channels
    constexpr uint32_t out_group_tiles = get_compile_time_arg_val(8);  // compute's push quantum
    [[maybe_unused]] constexpr bool stagger_row_walk = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t writer_window = get_compile_time_arg_val(10);  // max NoC1 reads outstanding when claiming
    constexpr uint32_t claim_chunk = get_compile_time_arg_val(11);    // sticks per claim (divides 32)
    constexpr uint32_t min_enter = get_compile_time_arg_val(12);  // enter an OPEN row only if >= this many unclaimed
    constexpr bool poll_in_flush = get_compile_time_arg_val(13) == 1;  // also serve claims while flushing writes
    constexpr bool gate_writes = get_compile_time_arg_val(14) == 1;    // count unacked output writes in the window
    constexpr auto q_args = TensorAccessorArgs<15>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto input_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr bool dyn_split = split_rows > 0;
    static_assert(split_rows <= TILE_DIM, "split_rows is a row count inside one tile-row");
    static_assert(claim_chunk > 0 && TILE_DIM % claim_chunk == 0, "claims tile the 32 body sticks");
    static_assert(writer_window >= claim_chunk, "a claim must fit the read window");

    uint32_t arg = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t input_col_offset_tiles = get_arg_val<uint32_t>(arg++);

    const auto q_acc = TensorAccessor(q_args, q_addr);
    const auto k_acc = TensorAccessor(k_args, k_addr);
    const auto v_acc = TensorAccessor(v_args, v_addr);

    // store_tile: global channel tile cg of tile-row r -> its q / k / v page.
    auto store_tile = [&](uint32_t src, uint32_t r, uint32_t cg) {
        if (cg < q_tiles) {
            GQKV_WRITE(src, q_acc.get_noc_addr(r * q_tiles + cg), tile_bytes);
        } else if (cg < q_tiles + k_tiles) {
            GQKV_WRITE(src, k_acc.get_noc_addr(r * k_tiles + (cg - q_tiles)), tile_bytes);
        } else {
            GQKV_WRITE(src, v_acc.get_noc_addr(r * v_tiles + (cg - q_tiles - k_tiles)), tile_bytes);
        }
    };

    // ---- dynamic split: shared words ----
    volatile tt_l1_ptr uint32_t* R = nullptr;
    volatile tt_l1_ptr uint32_t* D = nullptr;
    volatile tt_l1_ptr uint32_t* W = nullptr;
    volatile tt_l1_ptr uint32_t* N = nullptr;
    [[maybe_unused]] uint32_t handled = 0;         // last tile-row seq this kernel is finished with
    [[maybe_unused]] uint32_t sticks_claimed = 0;  // profiler evidence: this RISC's share of the input
    if constexpr (dyn_split) {
        cb_reserve_back(cb_split_done, 1);
        W = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_split_done));
        N = W + 1;
        *W = 0;
        *N = 0;
        cb_push_back(cb_split_done, 1);       // "W is initialized" (never popped: the page stays shared)
        cb_wait_front(cb_split_requests, 1);  // "R is initialized"
        R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_split_requests));
        D = R + 4;
    }

    // Serve at most one tile-row of the reader's (see the head comment). Returns at once when there
    // is nothing to claim; stops claiming as soon as output_ready() holds.
    [[maybe_unused]] auto poll = [&](auto&& output_ready) {
        if constexpr (dyn_split) {
            invalidate_l1_cache();
            uint32_t r = *R;
            const uint32_t seq = r >> 16;
            if (seq == handled) {
                return;
            }
            uint32_t st = r & 0xff;
            if (st == R_CLOSED) {
                handled = seq;
                return;
            }
            if (st == R_OPEN && TILE_DIM - ((r >> 8) & 0xff) < min_enter) {
                return;  // the reader is about to close it alone
            }
            const uint32_t dst = *D;
            *W = (seq << 16) | (TILE_DIM << 8) | W_ENTER;
            {
                MaybeDeviceZoneScope("writer_split_ack");
                while (st == R_OPEN) {
                    invalidate_l1_cache();
                    r = *R;
                    st = (r >> 16) == seq ? (r & 0xff) : R_CLOSED;
                }
            }
            handled = seq;
            if (st == R_CLOSED) {
                return;
            }
            const auto input_acc = TensorAccessor(input_args, input_addr);
            const uint32_t input_byte_offset =
                (input_col_offset_tiles + ch_group_tile_start) * (stick_bytes / block_ch_tiles);
            const uint32_t token0 = (core_seq_start + seq - 1) * TILE_DIM;
            uint32_t b = TILE_DIM;  // sticks [b, 32) claimed here
            {
                MaybeDeviceZoneScope("writer_split_issue");
                // Stop when the reader holds every stick below b, when it has left this tile-row (speculative
                // push: it re-read our share itself), or when our own output is ready.
                while ((r >> 16) == seq && b > ((r >> 8) & 0xff) && !output_ready()) {
                    uint32_t in_flight = reads_outstanding();
                    if constexpr (gate_writes) {
                        constexpr uint32_t write_weight = tile_bytes > stick_bytes ? tile_bytes / stick_bytes : 1;
                        in_flight += writes_unacked() * write_weight;  // NoC1 bytes in flight, in sticks
                    }
                    if (in_flight + claim_chunk <= writer_window) {
                        b -= claim_chunk;
                        *W = (seq << 16) | (b << 8) | W_CLAIM;
                        for (uint32_t c = 0; c < claim_chunk; ++c) {
                            GQKV_READ(
                                input_acc.get_noc_addr(token0 + b + c, input_byte_offset),
                                dst + (b + c) * stick_bytes,
                                stick_bytes);
                        }
                    }
                    invalidate_l1_cache();
                    r = *R;
                }
            }
            *N = noc_reads_num_issued[noc_index];  // the reader waits for this core's NoC1 responses to reach it
            *W = (seq << 16) | (b << 8) | W_DONE;
            sticks_claimed += TILE_DIM - b;
        }
    };

    for (uint32_t r = core_seq_start; r < core_seq_start + core_seq_tiles; ++r) {
        for (uint32_t c0 = 0; c0 < block_ch_tiles; c0 += out_group_tiles) {
            const uint32_t group = (block_ch_tiles - c0 < out_group_tiles) ? (block_ch_tiles - c0) : out_group_tiles;
            {
                MaybeDeviceZoneScope("writer_out_wait");
                if constexpr (dyn_split) {
                    auto ready = [&]() { return cb_pages_available_at_front(cb_output_tiles, group); };
                    while (!ready()) {
                        poll(ready);
                    }
                } else {
                    cb_wait_front(cb_output_tiles, group);
                }
            }
            const uint32_t src = get_read_ptr(cb_output_tiles);
            {
                MaybeDeviceZoneScope("writer_out_issue");
                for (uint32_t i = 0; i < group; ++i) {
                    store_tile(src + i * tile_bytes, r, ch_group_tile_start + c0 + i);
                }
            }
            {
                MaybeDeviceZoneScope("writer_out_flush");
                if constexpr (dyn_split && poll_in_flush) {
                    auto sent = [&]() { return ncrisc_noc_nonposted_writes_sent(noc_index); };
                    while (!sent()) {
                        poll(sent);
                    }
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_output_tiles, group);
        }
    }
    noc_async_write_barrier();
    noc_async_read_barrier();  // claims are never barriered in poll() (the reader tracks them)
#if defined(PROFILE_KERNEL)
    if constexpr (dyn_split) {
        DeviceTimestampedData("writer_split_sticks", sticks_claimed);
    }
#endif
}

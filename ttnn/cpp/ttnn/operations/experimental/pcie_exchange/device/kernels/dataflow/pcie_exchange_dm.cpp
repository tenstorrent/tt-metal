// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One core's share of a 2-device PCIe peer-to-peer exchange. Both devices run the same program; the peer's
// core with the same logical coordinates is this core's partner.
//
// Per-core exchange buffer (persistent, same address on both devices):
//   [landing 0 | landing 1 | control tile]   control: +0 flag written by the peer, +64 local sequence,
//                                            +128 staging word for the flag write.
// Protocol per invocation k (the sequence number lives in L1 so trace replays stay correct):
//   1. write my shard into the peer's landing[k & 1]            (PCIe, 27.9 GB/s)
//   2. write k into the peer's flag                              (same NoC path, lands after the payload)
//   3. wait until the peer's k arrived in my flag
//   4. ALL_REDUCE: hand input + landing to the compute kernel (CBs alias the buffers, nothing moves)
//      ALL_GATHER: NoC-copy my shard and the peer's shard to their output cores on this device
// Landing is double-buffered by parity: the peer only moves to k+2 after it saw my k+1, which I send after
// the program that consumed landing[k & 1] has finished on this core. The peer can therefore be at most one
// round ahead of me, never two.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr bool bfp8_payload = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_landing = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stage_dm = get_compile_time_arg_val(3);
    uint32_t i = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t xb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t shard_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t ntiles = get_arg_val<uint32_t>(i++);
    const uint32_t mode = get_arg_val<uint32_t>(i++);  // 0 = ALL_REDUCE, 1 = ALL_GATHER
    const uint32_t win_lo = get_arg_val<uint32_t>(i++);
    const uint32_t win_hi = get_arg_val<uint32_t>(i++);
    const uint32_t dst_a_x = get_arg_val<uint32_t>(i++);
    const uint32_t dst_a_y = get_arg_val<uint32_t>(i++);
    const uint32_t dst_b_x = get_arg_val<uint32_t>(i++);
    const uint32_t dst_b_y = get_arg_val<uint32_t>(i++);
    const uint32_t ctrl = get_arg_val<uint32_t>(i++);  // control tile: in the exchange shard or a separate tensor
    const uint32_t payload_bytes = get_arg_val<uint32_t>(i++);  // bytes shipped per round (bfp8: 1088 per tile)

    const uint64_t win = (static_cast<uint64_t>(win_hi) << 32) | win_lo;  // peer core L1 address 0, via PCIe
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl);
    volatile tt_l1_ptr uint32_t* local_seq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl + 64);
    volatile tt_l1_ptr uint32_t* staging = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl + 128);

    invalidate_l1_cache();
    const uint32_t seq = local_seq[0] + 1;
    local_seq[0] = seq;
    const uint32_t par = seq & 1;

    uint32_t src_addr = in_addr;
    if constexpr (bfp8_payload) {
        // Hand the bf16 shard to compute first; it packs the bfp8 copy we ship.
        cb_push_back(cb_in, ntiles);
        cb_wait_front(cb_stage_dm, ntiles);
        src_addr = get_read_ptr(cb_stage_dm);
    }
    noc_async_write(src_addr, win + xb_addr + par * payload_bytes, payload_bytes);
    staging[0] = seq;
    staging[1] = seq;
    staging[2] = seq;
    staging[3] = seq;
    noc_async_write(ctrl + 128, win + ctrl, 16);
    if (mode == 1) {
        noc_async_write(in_addr, get_noc_addr(dst_a_x, dst_a_y, out_addr), shard_bytes);
    }

    // Status word at ctrl+192 (host-readable through the exchange tensor): 1 = peer flag seen, 2 = timed out
    // (low half carries the flag value observed). A timeout still completes the program so the host can
    // read the diagnostics instead of hanging the device.
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl + 192);
    uint32_t spins = 0;
    uint32_t st = 1;
    // The peer may already be one round ahead (it only needs my flag for round k to start round k+1), in which
    // case its flag k+1 has overwritten k before I looked: accept any flag at or beyond seq (wraparound-safe).
    while (true) {
        invalidate_l1_cache();
        if (static_cast<int32_t>(flag[0] - seq) >= 0) {
            break;
        }
        if (++spins > 300000000u) {
            st = 0x20000 | (flag[0] & 0xffff);
            break;
        }
    }
    status[0] = st;
    status[1] = seq;
    status[2] = spins;

    if (mode == 0) {
        if constexpr (!bfp8_payload) {
            cb_push_back(cb_in, ntiles);
        }
        cb_push_back(cb_landing, 2 * ntiles);
    } else {
        noc_async_write(xb_addr + par * shard_bytes, get_noc_addr(dst_b_x, dst_b_y, out_addr), shard_bytes);
    }
    noc_async_write_barrier();
    if constexpr (bfp8_payload) {
        if (mode == 0) {
            cb_pop_front(cb_stage_dm, ntiles);  // the shipped copy may be overwritten only now
        }
    }
}

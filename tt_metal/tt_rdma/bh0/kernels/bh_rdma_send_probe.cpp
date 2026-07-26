// SPDX-License-Identifier: Apache-2.0
//
// BH.1 / M-1a — first TT-RDMA-v1 frame on the wire from an on-core RISC1 kernel.
//
// Builds a spec-compliant 0xFE PROBE frame (tt-rdma-wire-protocol-v1.md §2: a
// link-layer keepalive with loopback contents in the payload) and raw-L2 sends
// it to the BlueField-3 "tt" MAC at ethertype 0x1AF6, on a DEDICATED TXQ so it
// never touches base FW's TXQ0 link maintenance. RISC0 keeps yielding to base FW
// (the coexistence model proved by BH.0). Confirm on the BF3 with:
//   tcpdump -i <ttport> ether proto 0x1af6 -xx
//
// This is the send half only (M-1a). RX-classifier landing is M-1b. Reliability,
// MR checks, and the other opcodes come later — a PROBE needs none of them.
//
// HARD CONTRACT (tt_rdma_l1_layout.h): only touch the RDMA SW-L1 region; never
// write >= 0x70000 (base FW / boot_results -> bricks the link). No base-FW calls.

#include <cstdint>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"       // opcodes, ethertype, hdr struct
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_hdr_build.h"  // tt_rdma_build_hdr + crc32
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"  // HB/STOP/TX_BUF0 addresses
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_tx.h"     // tt_rdma_txpkt_config + send_raw

void kernel_main() {
    // arg0 = progress/heartbeat L1 addr (TT_RDMA_HB_ADDR): counts frames sent.
    // arg1 = stop-flag L1 addr (TT_RDMA_STOP_ADDR), or 0 to disable.
    // arg2 = num_frames: 0 = send until the stop flag; >0 = send exactly this many.
    // arg3 = spin iterations between frames (pacing so tcpdump / a poll can watch).
    // arg4 = dst MAC high 32 bits (mac >> 32); arg5 = dst MAC low 32 bits.
    // arg6 = payload byte length (PROBE loopback contents; 0 => 32 B default).
    const uint32_t hb_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_frames = get_arg_val<uint32_t>(2);
    const uint32_t spin = get_arg_val<uint32_t>(3);
    const uint32_t dmac_hi = get_arg_val<uint32_t>(4);
    const uint32_t dmac_lo = get_arg_val<uint32_t>(5);
    uint32_t payload_len = get_arg_val<uint32_t>(6);
    if (payload_len == 0) {
        payload_len = 32u;  // 32 B hdr + 32 B payload = 64 B >= Ethernet min, 16-B aligned
    }
    // arg7 = burst_bytes: if >0, BURST mode — one big raw transfer per command from the WQE region,
    //        which the HW auto-splits into <= arg8 (max_pkt) frames. This removes the per-frame
    //        register-write overhead (the ~650k-fps ceiling) and is the raw-mode bandwidth path.
    const uint32_t burst_bytes = get_arg_val<uint32_t>(7);
    const uint32_t max_pkt = get_arg_val<uint32_t>(8);

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;  // clear on entry (stale value must not stop us immediately)
    }

    // Configure our dedicated TXPKT row once: BF3 dest MAC + ethertype 0x1AF6.
    const uint64_t dst_mac = ((uint64_t)dmac_hi << 32) | (uint64_t)dmac_lo;
    tt_rdma_txpkt_config(TT_RDMA_TX_QUEUE, dst_mac, (uint16_t)TT_RDMA_ETHERTYPE);

    // ---- BURST mode: one big raw transfer per command; HW auto-splits into max_pkt frames. ----
    if (burst_bytes != 0) {
        tt_rdma_set_max_pkt_size(TT_RDMA_TX_QUEUE, max_pkt);
        // Fill the large source region (WQE payload, up to 128 KB) once; a spec PROBE header sits at
        // the very start so frame[0] is still recognizable on the wire.
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_WQE_PAYLOAD_ADDR);
        for (uint32_t i = 0; i < (burst_bytes >> 2); ++i) {
            src[i] = 0xAA55AA55u;
        }
        tt_rdma_hdr_t h0;
        tt_rdma_build_hdr(&h0, TT_OP_PROBE, TT_RDMA_VERSION, (uint16_t)0x50B, burst_bytes, 1u, 0u, 0u, 0u);
        const uint32_t* hw0 = reinterpret_cast<const uint32_t*>(&h0);
        for (uint32_t w = 0; w < (TT_RDMA_HDR_BYTES >> 2); ++w) {
            src[w] = hw0[w];
        }
        for (uint32_t burst = 1;; ++burst) {
            tt_rdma_send_raw(TT_RDMA_TX_QUEUE, TT_RDMA_WQE_PAYLOAD_ADDR, burst_bytes);
            *hb = burst;  // counts BURSTS (each = burst_bytes / max_pkt frames)
            if (num_frames != 0 && burst >= num_frames) {
                break;
            }
            if (stop != nullptr && *stop != 0) {
                break;
            }
            for (volatile uint32_t i = 0; i < spin; ++i) {
            }
        }
        return;
    }

    // The frame body lives in the RDMA TX buffer: [32-B header][payload].
    volatile tt_l1_ptr uint32_t* tx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_TX_BUF0_ADDR);
    const uint32_t frame_bytes = TT_RDMA_HDR_BYTES + payload_len;

    // Fill the loopback payload once (recognizable AA/55 pattern, spec-agnostic).
    for (uint32_t i = 0; i < (payload_len >> 2); ++i) {
        tx[(TT_RDMA_HDR_BYTES >> 2) + i] = 0xAA55AA55u;
    }

    for (uint32_t frame = 1;; ++frame) {
        // Build a fresh PROBE header each frame so seq advances (ver=1, no flags).
        tt_rdma_hdr_t h;
        tt_rdma_build_hdr(
            &h,
            TT_OP_PROBE,      // 0xFE
            TT_RDMA_VERSION,  // version_flags: ver=1, no IMM/REQ_ACK/SOLICITED
            (uint16_t)0x50B,  // tag: opaque PROBE cookie ("PrB")
            payload_len,      // length = payload bytes (not incl header)
            frame,            // seq
            0u,               // rkey unused for PROBE
            0u,               // remote_offset unused
            0u);              // imm unused
        // Copy the 32-B header into L1 as 8 words (packed struct is contiguous LE).
        const uint32_t* hw = reinterpret_cast<const uint32_t*>(&h);
        for (uint32_t w = 0; w < (TT_RDMA_HDR_BYTES >> 2); ++w) {
            tx[w] = hw[w];
        }

        tt_rdma_send_raw(TT_RDMA_TX_QUEUE, TT_RDMA_TX_BUF0_ADDR, frame_bytes);
        *hb = frame;  // observers watch this advance (== num_frames when bounded)

        if (num_frames != 0 && frame >= num_frames) {
            break;  // bounded send complete
        }
        if (stop != nullptr && *stop != 0) {
            break;  // host asked us to stop -> return so the go-loop reaps us (RISC1 idles)
        }
        for (volatile uint32_t i = 0; i < spin; ++i) {
            // pace only
        }
    }
}

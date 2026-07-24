// SPDX-License-Identifier: Apache-2.0
//
// BH.2-RX — TT-RDMA-v1 receive + opcode dispatch on RISC1 (tt-rdma-fw-arch-rx.md, BH variant).
//
// There is NO HW ethertype classifier on BH (eth_rx_flow.cpp is dead/uncompiled) — reception is via
// the base-FW dst-MAC router: unicast -> RXQ2, which base FW never drains (agent-confirmed). This
// kernel puts RXQ2 in raw mode (base FW leaves it in packet mode), so incoming frames land L2-stripped
// as [32B tt_rdma_hdr][payload], contiguous. It walks the landed bytes, parses each 32B header, and
// dispatches by opcode:
//   SEND/SEND_IMM (0x01/0x02) -> count (a real impl DMA-pushes to the host RxWqeRing)
//   WRITE/WRITE_IMM (0x10/0x11) -> MR-table lookup by rkey, bounds-check, land payload at
//                                  mr.base_noc_addr + remote_offset (Stage 1: base_noc is a LOCAL L1
//                                  byte address; Stage 2 swaps the copy for noc_async_write off-core)
//   other -> count as unknown
//
// Stage 1 framing model: RXQ2 in raw NOWRAP; a read pointer advances 0->BUF_PTR, processing each frame
// EXACTLY ONCE (no wrap, no reset -> no race). BUF_PTR is in BYTES (M-1b: it pegged at the 16KB ring
// size, not 16K words, which would have overrun 0x70000 and bricked the link). Total RX per run is
// bounded by the ring size; BUF_WRAP streaming is Stage 2.
//
// HARD CONTRACT (tt_rdma_l1_layout.h): only touch the RDMA SW-L1 region; never write >= 0x70000.

#include <cstdint>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"

// Copy n bytes L1->L1 as words (src/dst are 16-B aligned here). Stage 1 WRITE landing; Stage 2 -> NoC.
static inline void l1_copy_words(uint32_t dst_byte, uint32_t src_byte, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_byte);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_byte);
    const uint32_t nw = (nbytes + 3u) >> 2;
    for (uint32_t i = 0; i < nw; ++i) {
        d[i] = s[i];
    }
}

void kernel_main() {
    // arg0 = hb L1 addr (publish total frames dispatched)     -> TT_RDMA_HB_ADDR
    // arg1 = stop-flag L1 addr (0 disables)                   -> TT_RDMA_STOP_ADDR
    // arg2 = stats base L1 addr (8 u32 published)             -> RCB dbg region
    // arg3 = rx buffer L1 byte addr (frames land here)        -> TT_RDMA_RX_RING_ADDR
    // arg4 = rx buffer size in bytes                          -> TT_RDMA_RX_RING_SIZE
    // arg5 = MR table L1 byte addr                            -> TT_RDMA_MR_TABLE_ADDR
    const uint32_t hb_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t stats_addr = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf = get_arg_val<uint32_t>(3);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(4);
    const uint32_t mr_table = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);
    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    tt_rdma_rxq_init_raw(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size);  // raw NOWRAP; L2-stripped landing

    uint32_t read_off = 0;  // bytes consumed from the ring
    uint32_t n_send = 0, n_write = 0, n_write_ok = 0, n_unknown = 0, n_bad = 0, total = 0, last_op = 0;

    for (;;) {
        const uint32_t wp = tt_rdma_rxq_bufptr(TT_RDMA_RX_QUEUE);  // bytes written by HW (NOWRAP)

        while (read_off + TT_RDMA_HDR_BYTES <= wp) {
            // Header words at read_off (16-B aligned so word-aligned). Packed LE:
            //   w0=[op|ver<<8|tag<<16] w1=length w2=seq w3=rkey w4=roff_lo w5=roff_hi w6=imm w7=cksum
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rx_buf + read_off);
            const uint32_t op = w[0] & 0xFFu;
            const uint32_t len = w[1];
            const uint32_t rkey = w[3];
            const uint32_t roff = w[4];  // Stage 1: 32-bit offset is plenty for local targets

            if (len > TT_RDMA_MAX_PAYLOAD) {
                n_bad++;  // implausible length -> framing lost; stop walking this batch
                break;
            }
            uint32_t frame = TT_RDMA_HDR_BYTES + len;
            frame = (frame + 15u) & ~15u;  // 16-B aligned frame stride
            if (read_off + frame > wp) {
                break;  // frame not fully landed yet
            }

            if (op == TT_OP_SEND || op == TT_OP_SEND_IMM) {
                ++n_send;
            } else if (op == TT_OP_WRITE || op == TT_OP_WRITE_IMM) {
                ++n_write;
                const uint32_t slot = rkey >> 24;  // rkey = (slot<<24)|(rand16<<8)|gen
                if (slot < TT_RDMA_MR_SLOTS) {
                    volatile tt_l1_ptr uint32_t* mr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + slot * 32u);
                    const uint32_t mr_base = mr[0];    // base_noc_addr low (Stage 1: local L1 byte addr)
                    const uint32_t mr_len = mr[2];     // length low
                    const uint32_t mr_rkey = mr[4];    // rkey
                    const uint32_t mr_access = mr[5];  // access_flags
                    if (mr_rkey == rkey && (mr_access & TT_MR_REMOTE_WRITE) && (roff + len) <= mr_len) {
                        l1_copy_words(mr_base + roff, rx_buf + read_off + TT_RDMA_HDR_BYTES, len);
                        ++n_write_ok;
                    }
                }
            } else {
                ++n_unknown;
            }
            last_op = op;
            ++total;
            read_off += frame;
        }

        *hb = total;
        stats[0] = total;
        stats[1] = n_send;
        stats[2] = n_write;
        stats[3] = n_write_ok;
        stats[4] = n_unknown;
        stats[5] = n_bad;
        stats[6] = last_op;
        stats[7] = read_off;

        if (stop != nullptr && *stop != 0) {
            break;
        }
        for (volatile uint32_t i = 0; i < 20000u; ++i) {
            // pace the poll
        }
    }
}

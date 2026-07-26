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
//   WRITE/WRITE_IMM (0x10/0x11) -> MR-table lookup by rkey, REMOTE_WRITE + bounds check, land payload
//                                  at mr.base_noc_addr + remote_offset (Stage 1: base is a LOCAL L1
//                                  byte address; Stage 2b swaps the copy for noc_async_write off-core)
//   other -> count as unknown
//
// Framing model (arg6 = wrap):
//   wrap=0 (Stage 1): RXQ2 raw NOWRAP; read_pos advances 0->BUF_PTR once, each frame processed EXACTLY
//     once (no wrap/reset -> race-free). Total RX per run bounded by the ring size.
//   wrap=1 (Stage 2a): RXQ2 raw BUF_WRAP; HW wraps at buf end (BUF_PTR wraps 0..buf_size), read_pos
//     wraps mod buf_size -> CONTINUOUS RX beyond the ring size. The consumer must keep up or the HW
//     write pointer laps unread data (avail==buf_size is indistinguishable from empty -> poll often).
// BUF_PTR is in BYTES (M-1b: it pegged at the 16KB ring size, not 16K words -> would have overrun
// 0x70000 and bricked the link). buf_size MUST be a multiple of 16 (every word read stays aligned).
//
// HARD CONTRACT (tt_rdma_l1_layout.h): only touch the RDMA SW-L1 region; never write >= 0x70000.

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"  // noc_async_write, get_noc_addr, noc_async_write_barrier

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_hdr_build.h"  // tt_rdma_crc32 (header validation)

// Wrap-aware word read from the ring (off is 4-B aligned; buf_size % 16 == 0 keeps it in-bounds).
static inline uint32_t ring_rd(uint32_t buf, uint32_t off, uint32_t buf_size) {
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + (off % buf_size));
}

// Wrap-aware copy of nbytes from ring[src_off..] to a linear L1 dst (Stage-1/2a WRITE landing).
static inline void ring_copy(uint32_t dst_byte, uint32_t buf, uint32_t src_off, uint32_t buf_size, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_byte);
    const uint32_t nw = (nbytes + 3u) >> 2;
    for (uint32_t j = 0; j < nw; ++j) {
        d[j] = ring_rd(buf, src_off + 4u * j, buf_size);
    }
}

void kernel_main() {
    // arg0 = hb L1 addr (total frames dispatched)   arg1 = stop flag   arg2 = stats base (8 u32)
    // arg3 = rx buffer L1 byte addr   arg4 = rx buffer size bytes   arg5 = MR table L1 byte addr
    // arg6 = wrap (0 = NOWRAP Stage 1, 1 = BUF_WRAP Stage 2a)
    const uint32_t hb_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t stats_addr = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf = get_arg_val<uint32_t>(3);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(4);
    const uint32_t mr_table = get_arg_val<uint32_t>(5);
    const uint32_t wrap = get_arg_val<uint32_t>(6);
    // Stage 2b: off-core WRITE landing via noc_async_write. If noc_base != 0, WRITE lands at
    // get_noc_addr(noc_x, noc_y, noc_base + remote_offset) — a Tensix L1 / DRAM / another core, moved
    // by the NoC engine (RISC issues one descriptor, does NOT copy bytes). noc_base == 0 -> Stage 1/2a
    // local L1 copy (ring_copy). This is the RX analog of the TX "RISC arms, HW moves" result.
    const uint32_t noc_x = get_arg_val<uint32_t>(7);
    const uint32_t noc_y = get_arg_val<uint32_t>(8);
    const uint32_t noc_base = get_arg_val<uint32_t>(9);
    const bool use_noc = (noc_base != 0);
    // arg10 = crc_check: validate CRC-32 over header bytes [0..27] vs the header_cksum field; drop +
    // count mismatches. Correctness/integrity gate (Phase 1.1). Default on; a perf sweep may disable it
    // (the bit-serial CRC is ~224 ops/frame -- Phase 3 can table/HW-offload it).
    const uint32_t crc_check = get_arg_val<uint32_t>(10);

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);
    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    tt_rdma_rxq_init(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size, wrap);  // raw; L2-stripped landing

    uint32_t read_pos = 0;  // byte position in the ring (mod buf_size in wrap mode)
    uint32_t n_send = 0, n_write = 0, n_write_ok = 0, n_unknown = 0, n_bad = 0, n_crc_err = 0, total = 0, last_op = 0;

    for (;;) {
        const uint32_t wp = tt_rdma_rxq_bufptr(TT_RDMA_RX_QUEUE);  // bytes written by HW
        uint32_t avail = wrap ? ((wp + rx_buf_size - read_pos) % rx_buf_size) : (wp - read_pos);

        while (avail >= TT_RDMA_HDR_BYTES) {
            // Read the full 32B header (8 LE words) wrap-aware into a contiguous local copy.
            //   w0=[op|ver<<8|tag<<16] w1=length w3=rkey w4=roff_lo ... w7=header_cksum.
            uint32_t hw[8];
            for (uint32_t i = 0; i < 8u; ++i) {
                hw[i] = ring_rd(rx_buf, read_pos + 4u * i, rx_buf_size);
            }
            const uint32_t op = hw[0] & 0xFFu;
            const uint32_t len = hw[1];
            const uint32_t rkey = hw[3];
            const uint32_t roff = hw[4];  // 32-bit offset (Stage 1/2a)

            if (len > TT_RDMA_MAX_PAYLOAD) {
                // Implausible length => the producer lapped the consumer (ring overflow) and read_pos now
                // points at overwritten/garbage bytes. RESYNC to the current write head: drop the stale
                // span and resume from now. Graceful degradation under overload (process at the drain
                // rate, drop the excess) instead of spinning on garbage forever (catastrophic collapse).
                ++n_bad;
                read_pos = wp;
                break;
            }
            uint32_t frame = TT_RDMA_HDR_BYTES + len;
            frame = (frame + 15u) & ~15u;  // 16-B aligned stride
            if (frame > avail) {
                break;  // frame not fully landed yet
            }

            // Header integrity: CRC-32 (ETH-CTRL ICRC poly) over bytes [0..27] must equal header_cksum
            // (hw[7]). Drop + count mismatches (corruption / spoofing) -- never dispatch an unvalidated
            // header. SW fallback here; the ROCE_ICRC engine can offload this (see tt-rdma-rx-dispatch-spec).
            if (crc_check && tt_rdma_crc32(reinterpret_cast<const uint8_t*>(hw), 28u) != hw[7]) {
                ++n_crc_err;
                read_pos = wrap ? ((read_pos + frame) % rx_buf_size) : (read_pos + frame);
                avail -= frame;
                continue;
            }

            if (op == TT_OP_SEND || op == TT_OP_SEND_IMM) {
                ++n_send;
            } else if (op == TT_OP_WRITE || op == TT_OP_WRITE_IMM) {
                ++n_write;
                const uint32_t slot = rkey >> 24;  // rkey = (slot<<24)|(rand16<<8)|gen
                if (slot < TT_RDMA_MR_SLOTS) {
                    volatile tt_l1_ptr uint32_t* mr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + slot * 32u);
                    const uint32_t mr_base = mr[0];    // base_noc_addr low (Stage 1/2a: local L1 byte addr)
                    const uint32_t mr_len = mr[2];     // length low
                    const uint32_t mr_rkey = mr[4];    // rkey
                    const uint32_t mr_access = mr[5];  // access_flags
                    if (mr_rkey == rkey && (mr_access & TT_MR_REMOTE_WRITE) && (roff + len) <= mr_len) {
                        const uint32_t src_off = (read_pos + TT_RDMA_HDR_BYTES) % rx_buf_size;
                        if (use_noc) {
                            // NoC engine moves the payload off-core; RISC issues the descriptor only
                            // (no per-byte copy). Straddle-split at the ring wrap boundary.
                            if (src_off + len <= rx_buf_size) {
                                noc_async_write(rx_buf + src_off, get_noc_addr(noc_x, noc_y, noc_base + roff), len);
                            } else {
                                const uint32_t first = rx_buf_size - src_off;
                                noc_async_write(rx_buf + src_off, get_noc_addr(noc_x, noc_y, noc_base + roff), first);
                                noc_async_write(
                                    rx_buf, get_noc_addr(noc_x, noc_y, noc_base + roff + first), len - first);
                            }
                        } else {
                            ring_copy(mr_base + roff, rx_buf, src_off, rx_buf_size, len);
                        }
                        ++n_write_ok;
                    }
                }
            } else {
                ++n_unknown;
            }
            last_op = op;
            ++total;
            read_pos = wrap ? ((read_pos + frame) % rx_buf_size) : (read_pos + frame);
            avail -= frame;
        }

        if (use_noc) {
            noc_async_write_barrier();  // drain this batch's off-core writes (bounds outstanding NoC cmds)
        }

        *hb = total;
        stats[0] = total;
        stats[1] = n_send;
        stats[2] = n_write;
        stats[3] = n_write_ok;
        stats[4] = n_unknown;
        stats[5] = n_bad;
        stats[6] = n_crc_err;
        stats[7] = last_op;
        stats[8] = read_pos;

        if (stop != nullptr && *stop != 0) {
            break;
        }
        for (volatile uint32_t i = 0; i < 2000u; ++i) {
            // pace the poll (kept short in wrap mode so we drain before the HW laps the ring)
        }
    }
}

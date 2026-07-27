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
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_tx.h"     // tt_rdma_send_raw (READ_RESP egress)
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

// RxWqeRing slot layout (host-sdk §3): 64 slots x 1536 B; per-slot 32B header then payload.
constexpr uint32_t TT_RXWQE_SLOT_BYTES = 1536u;
constexpr uint32_t TT_RXWQE_PAYLOAD_OFF = 0x20u;
constexpr uint32_t TT_RXWQE_OWNED_BY_HOST = (1u << 8);  // flags byte at slot+0x14, bit0 (word bit8)

// Publish one RxWqeRing completion slot (host-sdk §3) to a NoC target and bump the producer index the
// host polls. `length` is the IN-SLOT payload length: SEND passes its payload len (copied into the slot
// from the ring at src_off, straddle-split at the wrap); WRITE_IMM passes 0 (payload already landed at
// the MR — the slot is completion-only, carrying `imm`). Header + prod_idx are staged in TX_BUF0 L1
// scratch because noc_async_write's source must be RDMA-L1, not the RISC stack (proven on silicon).
static inline void rxwqe_publish(
    uint32_t sr_x,
    uint32_t sr_y,
    uint32_t sr_base,
    uint32_t sr_slots,
    uint32_t sr_prodidx,
    uint32_t* sr_prod,
    uint32_t peer_seq,
    uint32_t length,
    uint32_t opcode,
    uint32_t imm,
    uint32_t cookie,
    uint32_t mr_idx,
    uint32_t rx_buf,
    uint32_t src_off,
    uint32_t rx_buf_size) {
    const uint32_t slot = *sr_prod % sr_slots;
    const uint32_t slot_base = sr_base + slot * TT_RXWQE_SLOT_BYTES;
    if (length > 0) {  // payload -> slot+0x20 (SEND); straddle-split at the ring wrap
        const uint32_t pay_dst = slot_base + TT_RXWQE_PAYLOAD_OFF;
        if (src_off + length <= rx_buf_size) {
            noc_async_write(rx_buf + src_off, get_noc_addr(sr_x, sr_y, pay_dst), length);
        } else {
            const uint32_t first = rx_buf_size - src_off;
            noc_async_write(rx_buf + src_off, get_noc_addr(sr_x, sr_y, pay_dst), first);
            noc_async_write(rx_buf, get_noc_addr(sr_x, sr_y, pay_dst + first), length - first);
        }
    }
    noc_async_write_barrier();  // payload commits before header (host-sdk: OWNED written last)
    volatile tt_l1_ptr uint32_t* sh = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_TX_BUF0_ADDR);
    sh[0] = peer_seq;                                   // +0x00 peer_seq
    sh[1] = length;                                     // +0x04 length (0 for WRITE_IMM)
    sh[2] = (opcode & 0xFFu);                           // +0x08 opcode | status(0)<<8
    sh[3] = imm;                                        // +0x0C immediate
    sh[4] = cookie;                                     // +0x10 cookie <- tag
    sh[5] = (mr_idx & 0xFFu) | TT_RXWQE_OWNED_BY_HOST;  // +0x14 mr_table_idx | flags=OWNED_BY_HOST
    sh[6] = 0u;
    sh[7] = 0u;
    noc_async_write(TT_RDMA_TX_BUF0_ADDR, get_noc_addr(sr_x, sr_y, slot_base), 32u);
    noc_async_write_barrier();
    ++(*sr_prod);  // bump producer index (host polls this = the completion)
    volatile tt_l1_ptr uint32_t* pi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_TX_BUF0_ADDR + 0x40u);
    *pi = *sr_prod;
    noc_async_write(TT_RDMA_TX_BUF0_ADDR + 0x40u, get_noc_addr(sr_x, sr_y, sr_prodidx), 4u);
    noc_async_write_barrier();
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
    // Phase 1.2a: SEND / SEND_IMM -> RxWqeRing publish. When send_ring_base != 0, a SEND lands as a
    // ring slot (host-sdk §3 format: 32B slot header + payload) at get_noc_addr(sr_x, sr_y, ring_base +
    // slot*SLOT_BYTES), slot = prod_idx % send_ring_slots, then the producer index at sr_prodidx is
    // bumped (host polls it — the completion). The productized target is a host hugepage (NoC->PCIe);
    // this first cut lands on a NoC-addressable core (Tensix L1) to prove byte-exact SEND delivery +
    // completion. send_ring_base == 0 -> SEND is only counted (unchanged; keeps the WRITE/streaming tests).
    const uint32_t sr_x = get_arg_val<uint32_t>(11);
    const uint32_t sr_y = get_arg_val<uint32_t>(12);
    const uint32_t sr_base = get_arg_val<uint32_t>(13);
    const uint32_t sr_slots = get_arg_val<uint32_t>(14);
    const uint32_t sr_prodidx = get_arg_val<uint32_t>(15);
    const bool use_send_ring = (sr_base != 0);
    uint32_t sr_prod = 0;
    uint32_t n_write_imm = 0;  // Phase 1.5: WRITE_IMM frames that raised an imm completion
    // Phase 1.6 access-control drop counters (each = an unauthorized WRITE provably NOT landed).
    uint32_t n_rkey_miss = 0, n_rkey_access = 0, n_rkey_bounds = 0;
    // Phase 1.3: READ target. When read_enable, a READ_REQ (0x20, header-only on the wire; the length
    // field is the request size) triggers: MR lookup (REMOTE_READ + bounds) -> noc_async_read the bytes
    // from the read source (rd_src_x,y,base + remote_offset) into a TX scratch buffer -> build a
    // READ_RESP (0x21) header (tag echoed for initiator correlation) -> tt_rdma_send_raw back to the
    // initiator (dst_mac). READ egress uses TXQ2 (separate block from RXQ2). read_enable == 0 -> READ_REQ
    // is only counted (unchanged; keeps the WRITE/SEND tests).
    const uint32_t read_enable = get_arg_val<uint32_t>(16);
    const uint32_t dst_mac_hi = get_arg_val<uint32_t>(17);
    const uint32_t dst_mac_lo = get_arg_val<uint32_t>(18);
    const uint32_t rd_src_x = get_arg_val<uint32_t>(19);
    const uint32_t rd_src_y = get_arg_val<uint32_t>(20);
    const uint32_t rd_src_base = get_arg_val<uint32_t>(21);
    uint32_t n_read_req = 0, n_read_resp = 0;
    // Phase 1.4: ACK (0x40) reception + cumulative-ACK accounting. An inbound ACK carries the peer's
    // cumulative ack_seq in the seq field (header-only frame). We track the highest ack_seq seen with
    // wraparound-safe signed comparison -- the "acked up to" watermark the TX/initiator side reads to
    // free retransmit buffers + complete sends. Monotonic: stale / duplicate / reordered ACKs are ignored.
    uint32_t n_ack = 0, ack_watermark = 0;

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);
    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    tt_rdma_rxq_init(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size, wrap);  // raw; L2-stripped landing
    if (read_enable) {  // configure TXQ2 raw egress once (static dst MAC + 0x1AF6) for READ_RESP frames
        const uint64_t dst_mac = ((uint64_t)dst_mac_hi << 32) | (uint64_t)dst_mac_lo;
        tt_rdma_txpkt_config(TT_RDMA_TX_QUEUE, dst_mac, (uint16_t)TT_RDMA_ETHERTYPE);
        tt_rdma_set_max_pkt_size(TT_RDMA_TX_QUEUE, TT_RDMA_MAX_PAYLOAD + TT_RDMA_HDR_BYTES);
    }

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
            // READ_REQ / ACK are HEADER-ONLY on the wire (wire-protocol §1: "payload absent for
            // READ_REQ/ACK") — their length field is semantic (request_len / ack_seq), not payload
            // present. So the on-wire frame stride excludes the payload for those opcodes.
            const bool hdr_only = (op == TT_OP_READ_REQ || op == TT_OP_ACK);
            uint32_t frame;
            if (hdr_only) {
                frame = TT_RDMA_HDR_ONLY_BYTES;  // fixed 48B (runt-pad-safe, 16-aligned); len is semantic
            } else {
                frame = TT_RDMA_HDR_BYTES + len;
                frame = (frame + 15u) & ~15u;  // 16-B aligned stride
            }
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
                if (use_send_ring) {
                    // Publish this SEND as one RxWqeRing slot (host-sdk §3) — payload rides in the slot.
                    const uint32_t src_off = (read_pos + TT_RDMA_HDR_BYTES) % rx_buf_size;
                    rxwqe_publish(
                        sr_x,
                        sr_y,
                        sr_base,
                        sr_slots,
                        sr_prodidx,
                        &sr_prod,
                        /*peer_seq=*/hw[2],
                        /*length=*/len,
                        /*opcode=*/op,
                        /*imm=*/hw[6],
                        /*cookie=*/(hw[0] >> 16) & 0xFFFFu,
                        /*mr_idx=*/0xFFu,
                        rx_buf,
                        src_off,
                        rx_buf_size);
                    ++n_write_ok;  // reuse write_ok as the "delivered to a NoC target" counter
                }
            } else if (op == TT_OP_WRITE || op == TT_OP_WRITE_IMM) {
                ++n_write;
                const uint32_t slot = rkey >> 24;  // rkey = (slot<<24)|(rand16<<8)|gen
                // Access-control enforcement (security-critical, Phase 1.6): resolve rkey -> MR, then check
                // access + bounds. Drop + count each failure class separately; NEVER land an unauthorized
                // WRITE. mr[4]=rkey (incl. generation byte), mr[5]=access_flags, mr[2]=length, mr[0]=base.
                volatile tt_l1_ptr uint32_t* mr =
                    (slot < TT_RDMA_MR_SLOTS) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + slot * 32u)
                                              : nullptr;
                if (mr == nullptr || mr[4] != rkey) {
                    ++n_rkey_miss;  // slot out of range, or no MR / rkey (incl. generation) mismatch
                } else if (!(mr[5] & TT_MR_REMOTE_WRITE)) {
                    ++n_rkey_access;  // MR exists but is not remotely writable
                } else if ((roff + len) > mr[2]) {
                    ++n_rkey_bounds;  // write would exceed the region
                } else {
                    const uint32_t mr_base = mr[0];
                    const uint32_t src_off = (read_pos + TT_RDMA_HDR_BYTES) % rx_buf_size;
                    if (use_noc) {
                        // NoC engine moves the payload off-core; RISC issues the descriptor only
                        // (no per-byte copy). Straddle-split at the ring wrap boundary.
                        if (src_off + len <= rx_buf_size) {
                            noc_async_write(rx_buf + src_off, get_noc_addr(noc_x, noc_y, noc_base + roff), len);
                        } else {
                            const uint32_t first = rx_buf_size - src_off;
                            noc_async_write(rx_buf + src_off, get_noc_addr(noc_x, noc_y, noc_base + roff), first);
                            noc_async_write(rx_buf, get_noc_addr(noc_x, noc_y, noc_base + roff + first), len - first);
                        }
                    } else {
                        ring_copy(mr_base + roff, rx_buf, src_off, rx_buf_size, len);
                    }
                    ++n_write_ok;
                    // Phase 1.5: WRITE_IMM (0x11) additionally raises a receiver completion carrying
                    // imm_data. The payload already landed at the MR; the completion is a length-0
                    // RxWqeRing slot with mr_table_idx = the target MR slot (host-sdk §3).
                    if (op == TT_OP_WRITE_IMM && use_send_ring) {
                        if (use_noc) {
                            noc_async_write_barrier();  // ensure the off-core WRITE landed before the completion
                        }
                        rxwqe_publish(
                            sr_x,
                            sr_y,
                            sr_base,
                            sr_slots,
                            sr_prodidx,
                            &sr_prod,
                            /*peer_seq=*/hw[2],
                            /*length=*/0u,
                            /*opcode=*/op,
                            /*imm=*/hw[6],
                            /*cookie=*/(hw[0] >> 16) & 0xFFFFu,
                            /*mr_idx=*/slot,
                            rx_buf,
                            0u,
                            rx_buf_size);
                        ++n_write_imm;
                    }
                }
            } else if (op == TT_OP_READ_REQ) {
                ++n_read_req;
                // READ target: fetch len bytes from the MR's read source and send them back as READ_RESP.
                // `len` is the request size (header-only frame). tag is echoed for initiator correlation.
                if (read_enable) {
                    const uint32_t slot = rkey >> 24;
                    if (slot < TT_RDMA_MR_SLOTS && len <= TT_RDMA_MAX_PAYLOAD) {
                        volatile tt_l1_ptr uint32_t* mr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + slot * 32u);
                        const uint32_t mr_len = mr[2];
                        const uint32_t mr_rkey = mr[4];
                        const uint32_t mr_access = mr[5];
                        if (mr_rkey == rkey && (mr_access & TT_MR_REMOTE_READ) && (roff + len) <= mr_len) {
                            const uint32_t tx = TT_RDMA_TX_BUF0_ADDR;  // [32B hdr][payload] staged in L1
                            // Fetch the requested bytes from the read source into the TX payload area.
                            noc_async_read(get_noc_addr(rd_src_x, rd_src_y, rd_src_base + roff), tx + 32u, len);
                            noc_async_read_barrier();
                            // Build the READ_RESP (0x21) header in place (tag echoed; length = fetched bytes).
                            volatile tt_l1_ptr uint32_t* rh = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tx);
                            const uint32_t tag = (hw[0] >> 16) & 0xFFFFu;
                            rh[0] = (uint32_t)TT_OP_READ_RESP | ((uint32_t)TT_RDMA_VERSION << 8) | (tag << 16);
                            rh[1] = len;    // payload length (present on the wire for READ_RESP)
                            rh[2] = hw[2];  // seq echoed
                            rh[3] = 0u;     // rkey unused on response
                            rh[4] = 0u;     // remote_offset unused
                            rh[5] = 0u;
                            rh[6] = 0u;  // imm
                            rh[7] = tt_rdma_crc32(reinterpret_cast<const uint8_t*>(tx), 28u);
                            tt_rdma_send_raw(TT_RDMA_TX_QUEUE, tx, TT_RDMA_HDR_BYTES + len);
                            ++n_read_resp;
                        }
                    }
                }
            } else if (op == TT_OP_ACK) {
                ++n_ack;
                const uint32_t ack_seq = hw[2];                // cumulative ack_seq rides in the seq field
                if ((int32_t)(ack_seq - ack_watermark) > 0) {  // wraparound-safe "newer than watermark"
                    ack_watermark = ack_seq;
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
        stats[9] = n_read_req;
        stats[10] = n_read_resp;
        stats[11] = n_ack;
        stats[12] = ack_watermark;
        stats[13] = n_write_imm;
        stats[14] = n_rkey_miss;
        stats[15] = n_rkey_access;
        stats[16] = n_rkey_bounds;

        if (stop != nullptr && *stop != 0) {
            break;
        }
        for (volatile uint32_t i = 0; i < 2000u; ++i) {
            // pace the poll (kept short in wrap mode so we drain before the HW laps the ring)
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
//
// BH.2a — RISC-off-datapath TX ring drainer (tt-rdma-tx-ring-spec.md).
//
// The RISC1 kernel ARMS pre-staged frames; it does NOT build headers or copy payload in the fast
// path (the producer/DMA staged them into the WQE payload region). Per WQE it does: read a 16-B
// descriptor, accept-ahead poll (ETH_TXQ_STATUS.CMD_ONGOING clears on ACCEPT, not drain), and 3 TXQ
// register writes (tt_rdma_send_raw). MAX_PKT auto-split fans each descriptor into many wire frames.
// This isolates the RISC arm-rate from the per-frame header-build/CRC/copy the M-1a probe paid.
//
// Descriptor ring @ TT_RDMA_WQE_DESCR_ADDR (64 x 16 B): {u32 frame_off, u32 frame_len, u32 flags_txq,
// u32 cookie}. The producer pre-fills payload slots + descriptors and publishes prod_idx (RCB). For
// a sustained arm-rate/BW measurement the kernel wraps and re-arms the ring_size descriptors.
//
// HARD CONTRACT (tt_rdma_l1_layout.h): only touch the RDMA SW-L1 region; never write >= 0x70000.

#include <cstdint>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_tx.h"

void kernel_main() {
    // arg0 = hb L1 addr (published arm count)           -> TT_RDMA_HB_ADDR
    // arg1 = stop-flag L1 addr (0 disables)             -> TT_RDMA_STOP_ADDR
    // arg2 = num_arms (0 = until stop; >0 = bounded)
    // arg3 = ring_size (# descriptors the producer staged)
    // arg4 = TXQ index; arg5 = max_pkt bytes (0 = HW default)
    // arg6 = dst MAC hi; arg7 = dst MAC lo
    const uint32_t hb_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_arms = get_arg_val<uint32_t>(2);
    const uint32_t ring_size = get_arg_val<uint32_t>(3);
    const uint32_t txq = get_arg_val<uint32_t>(4);
    const uint32_t max_pkt = get_arg_val<uint32_t>(5);
    const uint32_t dmac_hi = get_arg_val<uint32_t>(6);
    const uint32_t dmac_lo = get_arg_val<uint32_t>(7);
    // arg8 = pace: spin iterations between arms. Raw START_RAW has no deep accept-ahead FIFO, so
    // arming faster than the TXQ drains wedges it — pace to the sustainable rate.
    const uint32_t pace = get_arg_val<uint32_t>(8);
    // arg9 = payload base L1 addr (0 => TT_RDMA_WQE_PAYLOAD_ADDR). Lets us test which L1 region the
    // TXQ will actually transmit from (diagnostic for the WQE-region-doesn't-TX issue).
    uint32_t payload_base = get_arg_val<uint32_t>(9);
    if (payload_base == 0) {
        payload_base = TT_RDMA_WQE_PAYLOAD_ADDR;
    }

    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    // One-time TX config (control path, not per-frame).
    const uint64_t dst_mac = ((uint64_t)dmac_hi << 32) | (uint64_t)dmac_lo;
    tt_rdma_txpkt_config(txq, dst_mac, (uint16_t)TT_RDMA_ETHERTYPE);
    tt_rdma_set_max_pkt_size(txq, max_pkt);

    // Descriptor ring (16 B/entry = 4 u32).
    volatile tt_l1_ptr uint32_t* descr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_WQE_DESCR_ADDR);
    const uint32_t rs = (ring_size == 0) ? 1u : ring_size;

    // Snapshot the TXQ counters BEFORE any arm (host diffs against the AFTER snapshot).
    tt_rdma_txq_snapshot(txq, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_DBG_BEFORE_ADDR));

    uint32_t arms = 0;
    bool done = false;
    while (!done) {
        for (uint32_t slot = 0; slot < rs; ++slot) {
            const uint32_t frame_off = descr[slot * 4 + 0];
            const uint32_t frame_len = descr[slot * 4 + 1];
            // FAST PATH: no header build, no payload copy. Accept-ahead + arm.
            tt_rdma_send_raw(txq, payload_base + frame_off, frame_len);
            *hb = ++arms;
            if (num_arms != 0 && arms >= num_arms) {
                done = true;
                break;
            }
            if (stop != nullptr && *stop != 0) {
                done = true;
                break;
            }
            for (volatile uint32_t i = 0; i < pace; ++i) {
                // pace to the TXQ drain rate (raw mode has no deep accept-ahead)
            }
        }
    }

    // Snapshot the TXQ counters AFTER the run so the host can see whether accepted CMDs ever
    // START/END a packet (0 delta on PKT_START == command accepted but no packet ever started).
    tt_rdma_txq_snapshot(txq, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_DBG_AFTER_ADDR));
}

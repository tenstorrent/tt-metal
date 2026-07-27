// SPDX-License-Identifier: Apache-2.0
//
// Experiment 1 — RX ingress-ceiling go/no-go (tt-rdma-rx-linerate-research.md §5.1).
//
// Question: can a Blackhole eth core's MAC land inbound 0x1AF6 frames into its L1 ring at line rate,
// with the RISC OFF the per-frame path? This kernel does ZERO per-frame work — it configures RXQ2 in
// raw BUF_WRAP mode and then only snapshots the raw MAC RX counters to L1 for the host to sample. The
// MAC writes L1 independently; in wrap mode there is no consumer-aware flow control (the SW read pointer
// is not a HW register), so the only reason the MAC would DROP is its own RX-buffer AFIFO filling up —
// i.e. the L1 write path not absorbing the wire. So:
//   drop == 0 and WORD_CNT advancing at ~wire rate  -> ingress absorbs line rate; the whole RX gap is
//     RISC control-plane -> a Tensix-drainer-pool architecture can reach 200G/link (pending the drain
//     half, experiment 2).
//   drop > 0 well below wire rate                    -> the MAC->L1 write side is itself the ceiling ->
//     single-link 200G RX is not reachable on this silicon without direct-to-MR HW landing.
//
// Reads only; touches no payload; never writes >= 0x70000.

#include <cstdint>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"

void kernel_main() {
    // arg0 = stats base (L1)   arg1 = stop flag   arg2 = rx_buf byte addr   arg3 = rx_buf size bytes
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t rx_buf = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    tt_rdma_rxq_init(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size, /*wrap=*/1u);  // raw BUF_WRAP; MAC lands freely
    const uint32_t qb = TT_ETH_RXQ0_BASE + TT_RDMA_RX_QUEUE * TT_ETH_RXQ_STRIDE;

    uint32_t iters = 0;
    for (;;) {
        // Pure counter snapshot — the RISC does NO per-frame work. Publish continuously so the host can
        // sample the advance rate (WORD_CNT is 16-B words: slower 32b wrap than BYTE_CNT at line rate).
        stats[0] = TT_ETH_REG32(qb + TT_ETH_RXQ_WORD_CNT);
        stats[1] = TT_ETH_REG32(qb + TT_ETH_RXQ_BYTE_CNT);
        stats[2] = TT_ETH_REG32(qb + TT_ETH_RXQ_PKT_END_CNT);
        stats[3] = TT_ETH_REG32(qb + TT_ETH_RXQ_PACKET_DROP_CNT);
        stats[4] = TT_ETH_REG32(qb + TT_ETH_RXQ_BUF_PTR);
        stats[5] = TT_ETH_REG32(qb + TT_ETH_RXQ_OUTSTANDING_WR_CNT);
        stats[6] = TT_ETH_REG32(TT_ETH_RXPKT_BUF_P_STAT);  // ingress AFIFO fullness (backpressure signal)
        stats[7] = TT_ETH_REG32(qb + TT_ETH_RXQ_PKT_START_CNT);
        stats[8] = ++iters;
        (void)rx_buf_size;

        if (stop != nullptr && *stop != 0) {
            break;
        }
    }
}

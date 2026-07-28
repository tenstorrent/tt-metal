// SPDX-License-Identifier: Apache-2.0
//
// Phase 3.1e — eth RISC1 as the drainer pool's CONTROL PLANE (off the per-frame data path). In the pool
// architecture the Tensix workers do all per-frame RDMA work (claim -> read -> rkey->MR -> land ->
// complete); this eth kernel does NO per-frame work. It:
//   1. Configures RXQ2 raw BUF_WRAP (the MAC fills the L1 ring) and publishes the produce head PKT_END_CNT
//      (the monotonic frame count the workers bound their claim by) -- a cheap register snapshot.
//   2. Owns the SHARED MR table: on a host/gateway registration doorbell it writes the MR entry into the
//      shared table (its own L1) and bumps the generation counter, so the workers refresh their cache.
//      This is the "RISC1 registers, workers read" authority -- the control plane, not the data plane.
// (ACK / READ_RESP control-op handling reuses the Phase-1 dispatch logic; folded in as a follow-up. The
// point proven here: the eth RISC is the MR/produce-head authority and never touches the WRITE flood.)

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"  // noc_async_write + get_noc_addr (push the head to workers)
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"

void kernel_main() {
    // arg0 stats base (L1)  arg1 stop flag  arg2 rx_buf byte addr  arg3 rx_buf size bytes
    // arg4 shared MR table L1 addr (this core)  arg5 registration-request L1 addr (this core)
    // arg6 num workers  arg7 head L1 addr on EACH worker  arg8+2i worker[i] NoC x  arg9+2i worker[i] NoC y
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t rx_buf = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(3);
    const uint32_t mr_table = get_arg_val<uint32_t>(4);
    const uint32_t reg_req = get_arg_val<uint32_t>(5);
    const uint32_t nworkers = get_arg_val<uint32_t>(6);
    const uint32_t head_local = get_arg_val<uint32_t>(7);  // where on each worker to write the produce head
    // Cache the worker NoC coords (max 16) so we don't re-read args every loop iteration.
    uint32_t wx[16], wy[16];
    const uint32_t nw = (nworkers < 16u) ? nworkers : 16u;
    for (uint32_t w = 0; w < nw; ++w) {
        wx[w] = get_arg_val<uint32_t>(8u + 2u * w);
        wy[w] = get_arg_val<uint32_t>(9u + 2u * w);
    }

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }
    // Registration request: [go, slot, base, len, rkey, access, dest_x, dest_y]. Gen counter = MR slot 63
    // word 0 (workers refresh on a bump). The control plane fulfils a request then clears `go`.
    volatile tt_l1_ptr uint32_t* rr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reg_req);
    volatile tt_l1_ptr uint32_t* gen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + 63u * 32u);

    tt_rdma_rxq_init(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size, /*wrap=*/1u);  // raw BUF_WRAP; MAC lands freely
    const uint32_t qb = TT_ETH_RXQ0_BASE + TT_RDMA_RX_QUEUE * TT_ETH_RXQ_STRIDE;
    // Head push-source: staged in RDMA-L1 (TX_BUF0), NOT the stats region. stats lives in the RCB/DBG
    // region, which is base-FW-owned: a RISC C++ store there is host-visible via NoC, but a noc_async_write
    // that READS it as SOURCE returns stale (same failure that made writes to RCB/DBG not reach workers ->
    // the gen counter had to move to MR slot 63). TX_BUF0 is a proven-good noc_async_write source.
    volatile tt_l1_ptr uint32_t* head_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TT_RDMA_TX_BUF0_ADDR);

    uint32_t iters = 0, n_reg = 0;
    for (;;) {
        // Publish the produce head (+ diagnostics) -- the workers read stats[2] = PKT_END_CNT.
        const uint32_t pkt_end = TT_ETH_REG32(qb + TT_ETH_RXQ_PKT_END_CNT);  // produce head (single read)
        stats[0] = TT_ETH_REG32(qb + TT_ETH_RXQ_WORD_CNT);
        stats[1] = TT_ETH_REG32(qb + TT_ETH_RXQ_BYTE_CNT);
        stats[2] = pkt_end;  // host-visible produce head (stats region is NoC-readable for the host)
        stats[3] = TT_ETH_REG32(qb + TT_ETH_RXQ_PACKET_DROP_CNT);
        stats[4] = TT_ETH_REG32(qb + TT_ETH_RXQ_BUF_PTR);
        stats[8] = ++iters;
        stats[9] = n_reg;  // MR registrations fulfilled by the control plane

        // 3.1f: PUSH the produce head to each worker's LOCAL L1 so workers read it locally (no NoC read of
        // this eth core). Stage in TX_BUF0 (RDMA-L1) -- the noc_async_write SOURCE must be a proven-good
        // RDMA-L1 slot, NOT the RCB/DBG stats region (base-FW-owned: NoC source-reads there return stale).
        head_stage[0] = pkt_end;
        for (uint32_t w = 0; w < nw; ++w) {
            noc_async_write(TT_RDMA_TX_BUF0_ADDR, get_noc_addr(wx[w], wy[w], head_local), 4u);
        }
        noc_async_write_barrier();

        // Control op: MR registration. On the doorbell, write the MR entry into the shared table and bump
        // the generation so every worker refreshes its cache -- RISC1 is the registration authority.
        if (rr[0] != 0u) {
            const uint32_t slot = rr[1];
            if (slot < 62u) {  // slots 62 (reg req) + 63 (gen) are reserved
                volatile tt_l1_ptr uint32_t* mr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + slot * 32u);
                mr[0] = rr[2];         // base
                mr[2] = rr[3];         // length
                mr[4] = rr[4];         // rkey
                mr[5] = rr[5];         // access
                mr[6] = rr[6];         // dest NoC x
                mr[7] = rr[7];         // dest NoC y
                gen[0] = gen[0] + 1u;  // bump generation -> workers refresh
                ++n_reg;
            }
            rr[0] = 0u;  // clear the doorbell
        }

        if (stop != nullptr && *stop != 0) {
            break;
        }
    }
    stats[9] = n_reg;
}

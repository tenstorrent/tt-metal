// SPDX-License-Identifier: Apache-2.0
//
// BH.1 / M-1b — receive a TT-RDMA-v1 frame FROM the BlueField-3 on an on-core RISC1 kernel.
//
// Puts RXQ2 into raw-L2 mode landing frames at TT_RDMA_RX_RING_ADDR, then polls the RXQ write
// pointer. When a 0x1af6 frame arrives from the BF3 (sent UNICAST so the MAC router steers it to
// RXQ2), BUF_PTR advances; the kernel publishes diagnostics to observable L1 so the host can read
// them: words-received, per-queue drop counters, and the first RX words (to byte-match the frame).
// Coexists with base FW on RISC0 (BH.0 model); base FW doesn't consume RXQ2 on a NIC/EXTERNAL core.
//
// HARD CONTRACT (tt_rdma_l1_layout.h): only touch the RDMA SW-L1 region; never write >= 0x70000.

#include <cstdint>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"  // HB/STOP/RX_RING addresses
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"     // tt_rdma_rxq_init_raw + poll

void kernel_main() {
    // arg0 = rx_words L1 addr (publish RXQ2 BUF_PTR here)     -> TT_RDMA_HB_ADDR
    // arg1 = stop-flag L1 addr, or 0 to disable               -> TT_RDMA_STOP_ADDR
    // arg2 = diag base L1 addr (8 words published)            -> a slot in the RCB region
    // arg3 = rx buffer L1 byte addr (frames land here)        -> TT_RDMA_RX_RING_ADDR
    // arg4 = rx buffer size in bytes                          -> TT_RDMA_RX_RING_SIZE
    const uint32_t rx_words_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t diag_addr = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf_addr = get_arg_val<uint32_t>(3);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(4);

    volatile tt_l1_ptr uint32_t* rx_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rx_words_addr);
    volatile tt_l1_ptr uint32_t* diag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(diag_addr);
    volatile tt_l1_ptr uint32_t* rxbuf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rx_buf_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    // Put our RDMA RX queue into raw mode landing at the RX ring.
    tt_rdma_rxq_init_raw(TT_RDMA_RX_QUEUE, rx_buf_addr, rx_buf_size);

    for (;;) {
        const uint32_t bp = tt_rdma_rxq_bufptr(TT_RDMA_RX_QUEUE);  // words received so far
        *rx_words = bp;
        // Diagnostics: [0]=rxq2 bufptr, [1..3]=drop counters for RXQ2/0/1 (where did frames go?),
        // [4..7]=first 4 RX words (byte-match the received frame; L2 may or may not be stripped).
        diag[0] = bp;
        diag[1] = tt_rdma_rxq_dropcnt(TT_RDMA_RX_QUEUE);
        diag[2] = tt_rdma_rxq_dropcnt(0);
        diag[3] = tt_rdma_rxq_dropcnt(1);
        diag[4] = rxbuf[0];
        diag[5] = rxbuf[1];
        diag[6] = rxbuf[2];
        diag[7] = rxbuf[3];

        if (stop != nullptr && *stop != 0) {
            break;  // host asked us to stop -> return so the go-loop reaps us (RISC1 idles)
        }
        for (volatile uint32_t i = 0; i < 50000u; ++i) {
            // pace the poll
        }
    }
}

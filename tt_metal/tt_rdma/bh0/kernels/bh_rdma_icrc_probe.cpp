// SPDX-License-Identifier: Apache-2.0
//
// BH ETH-CTRL ROCE_ICRC characterization probe (RISC1). Answers the open question from
// tt-rdma-rx-dispatch-spec §9 / production-plan 1.1 follow-up: can the inline ROCE_ICRC
// hardware CRC engine (regs @ 0xFFB98000, tt_rdma_eth_icrc.h) offload the TT-RDMA header
// CRC off the RISC hot path? The wire header_cksum was moved onto the engine's polynomial
// (CRC-32, 0x04C11DB7) precisely so this can work.
//
// It does NOT dispatch or enforce anything — it observes. Sequence:
//   1. snapshot the POR ICRC config (CTRL / RX_INIT / TX_INIT) into stats,
//   2. optionally program CTRL and RX_INIT from args (sentinel 0xFFFFFFFF = leave as-is),
//   3. put RXQ2 in raw mode so known frames from the BF3 land AND the RX datapath (which the
//      engine snoops) is active,
//   4. on each landed frame: read the first 32B header, compute the SOFTWARE tt_rdma_crc32
//      over bytes [0..27], and snapshot RX_CALCULATED / RX_RECEIVED next to it.
// The host then compares: if RX_CALCULATED == sw_crc for some CTRL bit-order/init, the engine
// engages on our frames and that config is the one to bake into the RX kernel (then flip
// RX_CHECK_EN on). If RX_CALCULATED stays 0 / never matches across the sweep, the engine does
// not engage on raw 0x1AF6 framing and the offload needs RoCE-shaped frames (or we keep a
// software slice-by-4 table). Either way we learn it empirically instead of guessing.
//
// SAFETY: RX_CHECK_EN is left OFF unless the caller sets it in the CTRL arg — observe before
// enforce, so a mis-tuned engine can never drop live frames. EXTERNAL/raw rails don't run RoCE.

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_icrc.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_hdr_build.h"  // tt_rdma_crc32 (the SW oracle)

static inline uint32_t rd_word(uint32_t buf, uint32_t off, uint32_t buf_size) {
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + (off % buf_size));
}

void kernel_main() {
    // arg0 stats base   arg1 stop flag   arg2 rx_buf byte addr   arg3 rx_buf_size
    // arg4 icrc_ctrl to program (0xFFFFFFFF = leave POR)   arg5 icrc_rx_init to program (0xFFFFFFFF = leave)
    // arg6 wrap (raw RX mode: 0 NOWRAP, 1 BUF_WRAP)
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t rx_buf = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(3);
    const uint32_t prog_ctrl = get_arg_val<uint32_t>(4);
    const uint32_t prog_rx_init = get_arg_val<uint32_t>(5);
    const uint32_t wrap = get_arg_val<uint32_t>(6);

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    // 1. Snapshot POR config BEFORE touching anything.
    const uint32_t ctrl_por = tt_icrc_rd(TT_ICRC_CTRL);
    const uint32_t rx_init_por = tt_icrc_rd(TT_ICRC_RX_INIT);
    const uint32_t tx_init_por = tt_icrc_rd(TT_ICRC_TX_INIT);

    // 2. Optionally program.
    if (prog_ctrl != 0xFFFFFFFFu) {
        tt_icrc_wr(TT_ICRC_CTRL, prog_ctrl);
    }
    if (prog_rx_init != 0xFFFFFFFFu) {
        tt_icrc_wr(TT_ICRC_RX_INIT, prog_rx_init);
    }
    const uint32_t ctrl_rb = tt_icrc_rd(TT_ICRC_CTRL);

    // 3. Raw RX so frames land and the datapath the engine snoops is active.
    tt_rdma_rxq_init(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size, wrap);

    uint32_t iters = 0, frames_seen = 0, last_bufptr = 0;
    uint32_t sw_crc = 0, hdr_cksum_field = 0, rx_calc = 0, rx_recv = 0;

    for (;;) {
        const uint32_t wp = tt_rdma_rxq_bufptr(TT_RDMA_RX_QUEUE);
        // A new frame arrived if the write pointer advanced by at least a full header.
        const uint32_t avail = wrap ? ((wp + rx_buf_size - last_bufptr) % rx_buf_size) : (wp - last_bufptr);
        if (avail >= TT_RDMA_HDR_BYTES) {
            // Read the FIRST landed header (position last_bufptr) and compute the SW CRC over [0..27].
            uint32_t hw[8];
            for (uint32_t i = 0; i < 8u; ++i) {
                hw[i] = rd_word(rx_buf, last_bufptr + 4u * i, rx_buf_size);
            }
            sw_crc = tt_rdma_crc32(reinterpret_cast<const uint8_t*>(hw), 28u);
            hdr_cksum_field = hw[7];
            // Snapshot the engine's live result right after the frame landed.
            rx_calc = tt_icrc_rd(TT_ICRC_RX_CALCULATED);
            rx_recv = tt_icrc_rd(TT_ICRC_RX_RECEIVED);
            ++frames_seen;
            last_bufptr = wp;  // advance past everything landed so far (probe: one sample per poll)
        }

        stats[0] = frames_seen;
        stats[1] = ctrl_por;
        stats[2] = rx_init_por;
        stats[3] = tx_init_por;
        stats[4] = ctrl_rb;
        stats[5] = rx_calc;
        stats[6] = rx_recv;
        stats[7] = sw_crc;
        stats[8] = hdr_cksum_field;
        stats[9] = wp;
        stats[10] = tt_rdma_rxq_dropcnt(TT_RDMA_RX_QUEUE);
        stats[11] = ++iters;

        if (stop != nullptr && *stop != 0) {
            break;
        }
        for (volatile uint32_t i = 0; i < 4000u; ++i) {
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed output store (STORE_ROUTE): the writer (BRISC) sends each output tile on whichever NoC reaches its DRAM bank
// in fewer hops from this core, NoC1 (its own) or NoC0.
//
// With every output tile on NoC1, the DRAM-write bandwidth a core gets depends on its grid position: half of each
// core's tiles go to the DRAM column on its far side, across the chip through every core's traffic, and the cores
// farthest along NoC1 from the DRAM columns get the least (Blackhole p150b, 514x1024 32->128: kernel spans 427 us at
// grid x 1 to 598 us at x 15). On the torus NoC0 runs +x then +y and NoC1 -y then -x, so the far column is a few hops
// away on the other NoC. The NoC is chosen per (core, bank) once at kernel start: hops on each NoC from this core's
// physical coordinates to the bank's endpoint on that NoC (the endpoints' translated coordinates mapped to physical
// ones through this core's NoC0 ID translation tables). About 40-60% of each core's tiles go on NoC0.
//
// Measured (Blackhole p150b, 110 cores, bit-identical, us): 514x1024 32->128 597 -> 545 (kernel span max/p50 1.18 ->
// 1.12), 1026x2048 32->128 1825 -> 1757 (1.08 -> 1.05). A fixed share or a share that follows each NoC's unsent-write
// backlog did less (perf_experiments/store_noc_split).
//
// NoC0 writes use command buffer 2 (the reader's inline-write buffer; the reader issues no NoC0 write on the paths that
// enable this) and the NIU's NoC0 nonposted-write counters, which only this RISC advances during the kernel: the writer
// keeps its own targets for them (BRISC's firmware syncs only its own NoC's software counters). The reader re-syncs its
// NoC0 write counters from the NIU once the writer's last NoC0 write landed (STORE_SEM_DONE), so its end-of-kernel
// checks hold.
//
// Raw NoC (capability): noc_async_write / noc_async_write_page issue on BRISC's write command buffer, which on NoC0 is
// the reader's; no dataflow helper issues on a chosen command buffer of the other NoC.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace conv2d_3x3 {

struct RoutedStore {
    static constexpr uint32_t noc0 = 0;
    static constexpr uint32_t cmd_buf0 = 2;
    uint32_t sent0 = 0;   // NoC0 NIU nonposted-write "sent" count once every NoC0 write issued so far left L1
    uint32_t acked0 = 0;  // ... and its write-ack count once they all landed
    uint32_t mask0 = 0;   // bank b's tiles go on NoC0

    // Physical coordinate of a translated one (NoC0's ID translation tables: 6 five-bit entries a word).
    static FORCE_INLINE uint32_t phys(uint32_t table0, uint32_t id) {
        constexpr uint32_t entry_mask = (1u << NOC_TRANSLATE_ID_WIDTH) - 1;
        const uint32_t w = NOC_CMD_BUF_READ_REG(noc0, 0, NOC_CFG(table0 + id / NOC_TRANSLATE_TABLE_XY_SIZE));
        return (w >> ((id % NOC_TRANSLATE_TABLE_XY_SIZE) * NOC_TRANSLATE_ID_WIDTH)) & entry_mask;
    }

    template <typename Accessor>
    FORCE_INLINE void init(const Accessor& acc) {
        sent0 = NOC_STATUS_READ_REG(noc0, NIU_MST_NONPOSTED_WR_REQ_SENT);
        acked0 = NOC_STATUS_READ_REG(noc0, NIU_MST_WR_ACK_RECEIVED);
        while (!noc_cmd_buf_ready(noc0, cmd_buf0)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc0, cmd_buf0, NOC_RET_ADDR_MID, 0);
        const uint32_t me = NOC_CMD_BUF_READ_REG(noc0, 0, NOC_NODE_ID);
        const int32_t xs = me & NOC_NODE_ID_MASK;
        const int32_t ys = (me >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        constexpr int32_t W = noc_size_x;
        constexpr int32_t H = noc_size_y;
        for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
            // Page b sits in bank b.
            const uint32_t c0 = static_cast<uint32_t>(acc.get_noc_addr(b, 0, noc0) >> NOC_ADDR_COORD_SHIFT);
            const uint32_t c1 = static_cast<uint32_t>(acc.get_noc_addr(b, 0, 1) >> NOC_ADDR_COORD_SHIFT);
            const int32_t x0 = phys(NOC_X_ID_TRANSLATE_TABLE_0, c0 & NOC_NODE_ID_MASK);
            const int32_t y0 = phys(NOC_Y_ID_TRANSLATE_TABLE_0, (c0 >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK);
            const int32_t x1 = phys(NOC_X_ID_TRANSLATE_TABLE_0, c1 & NOC_NODE_ID_MASK);
            const int32_t y1 = phys(NOC_Y_ID_TRANSLATE_TABLE_0, (c1 >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK);
            const int32_t h0 = (x0 - xs + W) % W + (y0 - ys + H) % H;
            const int32_t h1 = (xs - x1 + W) % W + (ys - y1 + H) % H;
            if (h0 < h1) {
                mask0 |= 1u << b;
            }
        }
#ifdef STORE_NOC1_ROWS
        if ((STORE_NOC1_ROWS >> ys) & 1) {
            mask0 = 0;
        }
#endif
    }

    template <typename Accessor>
    FORCE_INLINE void write_page(uint32_t page, const Accessor& acc, uint32_t src) {
        if ((mask0 >> (page % NUM_DRAM_BANKS)) & 1) {
            while (!noc_cmd_buf_ready(noc0, cmd_buf0)) {
            }
            ncrisc_noc_fast_write<DM_DEDICATED_NOC, false, false>(
                noc0,
                cmd_buf0,
                src,
                acc.get_noc_addr(page, 0, noc0),
                acc.get_aligned_page_size(),
                NOC_UNICAST_WRITE_VC,
                false,
                false,
                1,
                false,
                false);
            ++sent0;
            ++acked0;
        } else {
            noc_async_write_page(page, acc, src);
        }
    }

    // Every write issued so far left L1 (both NoCs).
    FORCE_INLINE void flushed() {
        noc_async_writes_flushed();
        while (NOC_STATUS_READ_REG(noc0, NIU_MST_NONPOSTED_WR_REQ_SENT) != sent0) {
        }
    }

    // Every NoC0 write issued so far landed (NoC1: noc_async_write_barrier).
    FORCE_INLINE void barrier0() {
        while (NOC_STATUS_READ_REG(noc0, NIU_MST_WR_ACK_RECEIVED) != acked0) {
        }
    }
};

}  // namespace conv2d_3x3

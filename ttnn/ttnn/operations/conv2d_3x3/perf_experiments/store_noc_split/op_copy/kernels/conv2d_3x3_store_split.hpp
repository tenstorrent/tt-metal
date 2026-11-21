// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Output store over both NoCs (STORE_SPLIT): the writer (BRISC) issues each output tile on its own NoC1, or on NoC0.
// Where all cores store at once, NoC1 DRAM-write bandwidth depends on the writing core's grid position (some cores'
// NoC1 write queues back up while the others' drain); NoC0 routes the same writes over other links.
//
// Policies (STORE_SPLIT):
//   0  NoC1 only.
//   1  dynamic: a tile goes on NoC1 while NoC1 holds fewer than d1 issued-but-unsent writes and its command buffer is
//      free, else on NoC0 while NoC0 holds fewer than d0, else the writer polls both. Each core's NoC0 share follows
//      how fast its NoC1 queue drains at its position.
//   2  static: tile i of the core's stream goes on NoC0 when i % den < num.
//   3  route: a tile goes on the NoC with fewer hops from this core to its DRAM bank's endpoint on that NoC (NoC0
//      runs +x then +y, NoC1 -y then -x on the physical torus; coordinates through the NoC0 ID translation table).
//   4  route + dynamic: the shorter NoC while its backlog is below its limit, else the other one.
//   5  NoC1 while its backlog is below d1; a tile whose bank is closer on NoC0 (policy 3's rule) spills to NoC0 while
//      NoC0's backlog is below d0.
//
// NoC0 writes use command buffer 2 (the reader's inline-write buffer; the reader issues no NoC0 write on the paths that
// enable this) and the NIU's NoC0 nonposted-write counters, which only this RISC advances there: the writer keeps its
// own targets for them (BRISC's firmware syncs only its own NoC's software counters).
//
// Row-group pipeline (OutPipe, STORE_PIPE): up to `cap` row groups of the output CB issued and not yet popped; a group
// is popped once its writes left L1 (both NoCs' sent counters passed its last write), so the writes of the next group
// queue behind it instead of waiting for a flush.
//
// Raw NoC (capability): noc_async_write / noc_async_write_page issue on BRISC's write command buffer, which on NoC0 is
// the reader's; no dataflow helper issues on a chosen command buffer of the other NoC, or reports a NoC's unsent-write
// backlog without blocking.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef STORE_HK
#define STORE_HK 1
#endif
#ifndef STORE_HT
#define STORE_HT 1
#endif
#ifndef STORE_N1Y
#define STORE_N1Y 1
#endif
#ifndef STORE_FIRST_TILES
#define STORE_FIRST_TILES 0
#endif
#ifndef STORE_SKIP_TILES
#define STORE_SKIP_TILES 0
#endif

namespace conv2d_3x3 {

template <uint32_t policy, uint32_t d1, uint32_t d0, uint32_t num, uint32_t den>
struct DualNocStore {
    static constexpr uint32_t noc0 = 0;
    static constexpr uint32_t cmd_buf0 = 2;
    static constexpr bool uses_noc0 = policy != 0;
    uint32_t sent0 = 0;   // NoC0 NIU nonposted-write "sent" count once every NoC0 write issued so far left L1
    uint32_t acked0 = 0;  // ... and its write-ack count once they all landed
    uint32_t n0 = 0;
    uint32_t n_all = 0;
    uint32_t mask0 = 0;  // bank b's tiles prefer NoC0 (policies 3, 4)

    // Physical NoC0-frame coordinate of a translated one (NoC0's ID translation tables, 6 five-bit entries a word).
    static FORCE_INLINE uint32_t phys(uint32_t table0, uint32_t id) {
        const uint32_t w = NOC_CMD_BUF_READ_REG(noc0, 0, NOC_CFG(table0 + id / 6));
        return (w >> ((id % 6) * 5)) & 31;
    }

    template <typename Accessor>
    FORCE_INLINE void init_routes(const Accessor& acc) {
        if constexpr (policy >= 3) {
            const uint32_t me = NOC_CMD_BUF_READ_REG(noc0, 0, NOC_NODE_ID);
            const int32_t xs = me & 63;
            const int32_t ys = (me >> 6) & 63;
            constexpr int32_t W = noc_size_x;
            constexpr int32_t H = noc_size_y;
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
                const uint32_t c0 = static_cast<uint32_t>(acc.get_noc_addr(b, 0, noc0) >> NOC_ADDR_COORD_SHIFT);
                const uint32_t c1 = static_cast<uint32_t>(acc.get_noc_addr(b, 0, 1) >> NOC_ADDR_COORD_SHIFT);
                const int32_t x0 = phys(NOC_X_ID_TRANSLATE_TABLE_0, c0 & 63);
                const int32_t y0 = phys(NOC_Y_ID_TRANSLATE_TABLE_0, (c0 >> 6) & 63);
                const int32_t x1 = phys(NOC_X_ID_TRANSLATE_TABLE_0, c1 & 63);
                const int32_t y1 = phys(NOC_Y_ID_TRANSLATE_TABLE_0, (c1 >> 6) & 63);
                // NoC0: +x along this core's row, then +y down the bank's column (shared by every row's writes);
                // NoC1: -y along this core's column, then -x along the endpoint's row (shared by every column's).
                // The shared final leg counts STORE_HK times.
                const int32_t h0 = (x0 - xs + W) % W + STORE_HK * ((y0 - ys + H) % H);
                const int32_t h1 = STORE_N1Y ? (ys - y1 + H) % H + STORE_HK * ((xs - x1 + W) % W)
                                             : (xs - x1 + W) % W + STORE_HK * ((ys - y1 + H) % H);
                if (h0 + STORE_HT <= h1) {
                    mask0 |= 1u << b;
                }
            }
        }
    }

    FORCE_INLINE void init() {
        if constexpr (policy != 0) {
            sent0 = NOC_STATUS_READ_REG(noc0, NIU_MST_NONPOSTED_WR_REQ_SENT);
            acked0 = NOC_STATUS_READ_REG(noc0, NIU_MST_WR_ACK_RECEIVED);
            while (!noc_cmd_buf_ready(noc0, cmd_buf0)) {
            }
            NOC_CMD_BUF_WRITE_REG(noc0, cmd_buf0, NOC_RET_ADDR_MID, 0);
        }
    }

    FORCE_INLINE uint32_t backlog1() const {
        return noc_nonposted_writes_num_issued[noc_index] -
               NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    }
    FORCE_INLINE uint32_t backlog0() const { return sent0 - NOC_STATUS_READ_REG(noc0, NIU_MST_NONPOSTED_WR_REQ_SENT); }

    FORCE_INLINE void issue0(uint32_t src, uint64_t dst, uint32_t bytes) {
        while (!noc_cmd_buf_ready(noc0, cmd_buf0)) {
        }
        ncrisc_noc_fast_write<DM_DEDICATED_NOC, false, false>(
            noc0, cmd_buf0, src, dst, bytes, NOC_UNICAST_WRITE_VC, false, false, 1, false, false);
        ++sent0;
        ++acked0;
        ++n0;
    }

    // `spin` runs while both NoCs are at their backlog limit (dynamic policy).
    template <typename Accessor, typename Spin>
    FORCE_INLINE void write_page(uint32_t page, const Accessor& acc, uint32_t src, uint32_t bytes, Spin&& spin) {
        bool on0 = false;
        if constexpr (policy == 2) {
            on0 = n_all % den < num;
        } else if constexpr (policy == 3) {
            on0 = ((mask0 >> (page % NUM_DRAM_BANKS)) & 1) && (STORE_FIRST_TILES == 0 || n_all < STORE_FIRST_TILES) &&
                  n_all >= STORE_SKIP_TILES;
        } else if constexpr (policy == 4) {
            const bool pref0 = (mask0 >> (page % NUM_DRAM_BANKS)) & 1;
            while (true) {
                const bool ok1 = backlog1() < d1 && noc_cmd_buf_ready(noc_index, write_cmd_buf);
                const bool ok0 = backlog0() < d0 && noc_cmd_buf_ready(noc0, cmd_buf0);
                if (pref0 ? ok0 : ok1) {
                    on0 = pref0;
                    break;
                }
                if (pref0 ? ok1 : ok0) {
                    on0 = !pref0;
                    break;
                }
                spin();
            }
        } else if constexpr (policy == 5) {
            // NoC1 while it drains; a tile whose bank is closer on NoC0 spills there once NoC1 backs up.
            const bool can0 = (mask0 >> (page % NUM_DRAM_BANKS)) & 1;
            while (true) {
                if (backlog1() < d1 && noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
                    break;
                }
                if (can0 && backlog0() < d0 && noc_cmd_buf_ready(noc0, cmd_buf0)) {
                    on0 = true;
                    break;
                }
                spin();
            }
        } else if constexpr (policy == 1) {
            while (true) {
                if (backlog1() < d1 && noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
                    break;
                }
                if (backlog0() < d0 && noc_cmd_buf_ready(noc0, cmd_buf0)) {
                    on0 = true;
                    break;
                }
                spin();
            }
        }
        if (on0) {
            issue0(src, acc.get_noc_addr(page, 0, noc0), bytes);
        } else {
            noc_async_write_page(page, acc, src);
        }
        ++n_all;
    }

    // Every write issued so far left L1 (both NoCs).
    FORCE_INLINE void flushed() {
        noc_async_writes_flushed();
        if constexpr (policy != 0) {
            while (NOC_STATUS_READ_REG(noc0, NIU_MST_NONPOSTED_WR_REQ_SENT) != sent0) {
            }
        }
    }
    // Every NoC0 write issued so far landed (NoC1: noc_async_write_barrier).
    FORCE_INLINE void barrier0() {
        if constexpr (policy != 0) {
            while (NOC_STATUS_READ_REG(noc0, NIU_MST_WR_ACK_RECEIVED) != acked0) {
            }
        }
    }
};

template <uint32_t cb, uint32_t group_tiles, uint32_t cap, typename Store>
struct OutPipe {
    Store& st;
    uint32_t t1[cap];
    uint32_t t0[cap];
    uint32_t head = 0;
    uint32_t n = 0;

    FORCE_INLINE bool oldest_sent() const {
        const int32_t s1 =
            static_cast<int32_t>(NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) - t1[head]);
        if constexpr (!Store::uses_noc0) {
            return s1 >= 0;
        } else {
            const int32_t s0 =
                static_cast<int32_t>(NOC_STATUS_READ_REG(Store::noc0, NIU_MST_NONPOSTED_WR_REQ_SENT) - t0[head]);
            return s1 >= 0 && s0 >= 0;
        }
    }
    FORCE_INLINE void pop_oldest() {
        cb_pop_front(cb, group_tiles);
        head = head + 1 == cap ? 0 : head + 1;
        --n;
    }
    FORCE_INLINE void try_pop() {
        while (n > 0 && oldest_sent()) {
            pop_oldest();
        }
    }
    // L1 address of the next row group once compute pushed it (`idle` runs while waiting).
    template <typename Idle>
    FORCE_INLINE uint32_t next_group(Idle&& idle) {
        while (n == cap) {
            try_pop();
        }
        while (!cb_pages_available_at_front(cb, (n + 1) * group_tiles)) {
            try_pop();
            idle();
        }
        const auto& itf = get_local_cb_interface(cb);
        uint32_t l1 = itf.fifo_rd_ptr + n * group_tiles * itf.fifo_page_size;
        if (l1 >= itf.fifo_limit) {
            l1 -= itf.fifo_size;
        }
        return l1;
    }
    FORCE_INLINE void group_issued() {
        uint32_t slot = head + n;
        slot = slot >= cap ? slot - cap : slot;
        t1[slot] = noc_nonposted_writes_num_issued[noc_index];
        t0[slot] = st.sent0;
        ++n;
    }
    FORCE_INLINE void drain() {
        while (n > 0) {
            try_pop();
        }
    }
};

}  // namespace conv2d_3x3

// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The WH SDPA op's I/O kernel (wh_sdpa_op_common.h), on every core: zeroes the sync lines and meets the group, then
// posts the launch token (GO) for the TRISCs and the feed; reads each item's Q block into its staging as [d slice][64
// queries][16 d] (two items ahead), writes each item's O staging ([2 query tiles][8 d slices][32 queries][16 d]) to
// O's tiles, and returns each K/V block TRISC0 has unpacked to its reader (a CONSUMED increment).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "wh_sdpa_op_common.h"

namespace {

inline volatile uint32_t& line(uint32_t scratch, uint32_t l) {
    return *reinterpret_cast<volatile uint32_t*>(whs_line(scratch, l));
}

}  // namespace

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(WHS_A_SCRATCH);
    const uint32_t go = get_arg_val<uint32_t>(WHS_A_GO);
    const uint32_t q_tile_rows = get_arg_val<uint32_t>(WHS_A_Q_TILE_ROWS);
    const uint32_t bufs = get_arg_val<uint32_t>(WHS_A_BUFS);
    const uint32_t readers = get_arg_val<uint32_t>(WHS_A_READERS);
    const uint32_t group_cores = get_arg_val<uint32_t>(WHS_A_GROUP_CORES);
    const uint32_t rounds = get_arg_val<uint32_t>(WHS_A_ROUNDS);
    const uint32_t leader_x = get_arg_val<uint32_t>(WHS_A_LEADER_X), leader_y = get_arg_val<uint32_t>(WHS_A_LEADER_Y);
    const uint32_t barrier = get_semaphore(get_arg_val<uint32_t>(WHS_A_BARRIER_SEM));
    const bool leader = get_arg_val<uint32_t>(WHS_A_READER_INDEX) == 0u;
    const InterleavedAddrGen<true> q_gen = {.bank_base_address = get_arg_val<uint32_t>(WHS_A_Q), .page_size = WHS_TILE};
    const InterleavedAddrGen<true> o_gen = {.bank_base_address = get_arg_val<uint32_t>(WHS_A_O), .page_size = WHS_TILE};

    for (uint32_t l = 0; l < WHS_S_LINES; ++l) {
        if (l != WHS_S_GO) {
            line(scratch, l) = 0u;
        }
    }
    // This launch's valid key count, before counting in: the feed and TRISCs read it once GO is posted.
    uint32_t valid_keys = get_arg_val<uint32_t>(WHS_A_VALID_KEYS);
    const uint32_t valid_addr = get_arg_val<uint32_t>(WHS_A_VALID_ADDR);
    if (valid_addr != 0u) {
        const InterleavedAddrGen<true> n_gen = {.bank_base_address = valid_addr, .page_size = 32u};
        noc_async_read(n_gen.get_noc_addr(0u), scratch + WHS_VALID_STAGE, 32u);
        noc_async_read_barrier();
        valid_keys = *reinterpret_cast<volatile uint32_t*>(scratch + WHS_VALID_STAGE);
    }
    line(scratch, WHS_S_VALID) = valid_keys;
    const uint32_t blocks = whs_key_blocks(valid_keys);
    // Group barrier: every core counts in at the leader, which then posts GO to the group.
    noc_semaphore_inc(get_noc_addr(leader_x, leader_y, barrier), 1u);
    if (leader) {
        volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier);
        noc_semaphore_wait(count, group_cores);
        line(scratch, WHS_S_GO) = go;
        if (group_cores > 1u) {
            const uint32_t x0 = get_arg_val<uint32_t>(WHS_A_GROUP_X0), y0 = get_arg_val<uint32_t>(WHS_A_GROUP_Y0);
            const uint32_t x1 = get_arg_val<uint32_t>(WHS_A_GROUP_X1), y1 = get_arg_val<uint32_t>(WHS_A_GROUP_Y1);
            const uint32_t go_line = whs_line(scratch, WHS_S_GO);
            const uint64_t dst = noc_index == 1 ? get_noc_multicast_addr(x1, y1, x0, y0, go_line)
                                                : get_noc_multicast_addr(x0, y0, x1, y1, go_line);
            noc_semaphore_set_multicast_loopback_src(go_line, dst, group_cores);
            noc_async_write_barrier();
        }
    } else {
        while (line(scratch, WHS_S_GO) != go) {
        }
    }

    // This core's items: (head, query block) of its non-idle rounds.
    uint32_t items = 0;
    const bool feed_only = get_arg_val<uint32_t>(WHS_A_DEBUG) & WHS_DEBUG_FEED_ONLY;
    for (uint32_t r = 0; r < rounds && !feed_only; ++r) {
        if ((get_arg_val<uint32_t>(WHS_A_ROUND0 + r) & 0xFFFFu) != WHS_IDLE) {
            ++items;
        }
    }
    const auto item_round = [&](uint32_t k) -> uint32_t {
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t w = get_arg_val<uint32_t>(WHS_A_ROUND0 + r);
            if ((w & 0xFFFFu) != WHS_IDLE && k-- == 0u) {
                return w;
            }
        }
        return 0u;
    };
    const uint32_t total = rounds * blocks;
    // A reader whose casts this kernel makes (WHS_CAST_BY_IO): its blocks reader_index, + readers, ... in order, each
    // once the feed has landed it, on this kernel's NoC.
    const uint32_t debug = get_arg_val<uint32_t>(WHS_A_DEBUG);
    const uint32_t reader_index = get_arg_val<uint32_t>(WHS_A_READER_INDEX);
    const bool casts = (debug & WHS_CAST_BY_IO) && reader_index != WHS_IDLE && group_cores > 1u;
    const uint32_t my_blocks = casts && reader_index < total ? (total - reader_index + readers - 1u) / readers : 0u;
    const uint32_t gx0 = get_arg_val<uint32_t>(WHS_A_GROUP_X0), gy0 = get_arg_val<uint32_t>(WHS_A_GROUP_Y0);
    const uint32_t gx1 = get_arg_val<uint32_t>(WHS_A_GROUP_X1), gy1 = get_arg_val<uint32_t>(WHS_A_GROUP_Y1);
    const auto mcast = [&](uint32_t addr) -> uint64_t {
        return noc_index == 1 ? get_noc_multicast_addr(gx1, gy1, gx0, gy0, addr)
                              : get_noc_multicast_addr(gx0, gy0, gx1, gy1, addr);
    };
    const bool data_casts = !(debug & WHS_DEBUG_NO_CASTS);
    uint32_t cast_next = 0;
    uint32_t forwarded = 0, q_next = 0, o_next = 0;
    while (forwarded < total || o_next < items || cast_next < my_blocks) {
        while (cast_next < my_blocks && line(scratch, WHS_S_LANDED) > cast_next) {
            const uint32_t g = reader_index + cast_next * readers, b = g % bufs;
            const uint32_t buffer = scratch + WHS_KV + b * WHS_BLOCK;
            line(scratch, WHS_S_READY + b) = g / bufs + 1u;
            if (data_casts) {
                if (debug & WHS_DEBUG_NO_PATH_RESERVE) {
                    ncrisc_noc_fast_write_any_len_loopback_src<noc_mode>(
                        noc_index,
                        write_cmd_buf,
                        buffer,
                        mcast(buffer),
                        WHS_BLOCK,
                        NOC_MULTICAST_WRITE_VC,
                        true,
                        true,
                        group_cores,
                        false);
                } else {
                    noc_async_write_multicast_loopback_src(buffer, mcast(buffer), WHS_BLOCK, group_cores, true);
                }
            }
            const uint32_t ready_addr = whs_line(scratch, WHS_S_READY + b);
            noc_semaphore_set_multicast_loopback_src(ready_addr, mcast(ready_addr), group_cores);
            ++cast_next;
        }
        const uint32_t unpacked = line(scratch, WHS_S_UNPACKED);
        while (forwarded < unpacked) {
            const uint32_t reader = forwarded % readers;
            const uint32_t rx = get_arg_val<uint32_t>(WHS_A_READER0_X + 2u * reader);
            const uint32_t ry = get_arg_val<uint32_t>(WHS_A_READER0_Y + 2u * reader);
            noc_semaphore_inc(get_noc_addr(rx, ry, whs_line(scratch, WHS_S_CONSUMED + forwarded % bufs)), 1u);
            ++forwarded;
        }
        if (q_next < items && q_next < line(scratch, WHS_S_Q_TAKEN) + 2u) {
            const uint32_t w = item_round(q_next);
            const uint32_t head = w >> 16, qb = w & 0xFFFFu;
            const uint32_t stage = scratch + WHS_UNPACK0 + (WHS_W_Q + q_next % 2u) * WHS_W_BYTES;
            for (uint32_t t = 0; t < 2u; ++t) {
                const uint32_t tile_row = (head * q_tile_rows + 2u * qb + t) * 4u;
                for (uint32_t dc = 0; dc < 4u; ++dc) {
                    for (uint32_t fi = 0; fi < 4u; ++fi) {
                        const uint32_t rh = fi / 2u, ch = fi % 2u;
                        noc_async_read(
                            q_gen.get_noc_addr(tile_row + dc, fi * WHS_TFACE),
                            stage + (2u * dc + ch) * WHS_FACE + (2u * t + rh) * WHS_TFACE,
                            WHS_TFACE);
                    }
                }
            }
            noc_async_read_barrier();
            line(scratch, WHS_S_Q_LANDED) = ++q_next;
        }
        if (o_next < items && line(scratch, WHS_S_O_STAGED) > o_next) {
            const uint32_t w = item_round(o_next);
            const uint32_t head = w >> 16, qb = w & 0xFFFFu;
            const uint32_t stage = scratch + WHS_O_STAGE + o_next % 2u * WHS_O_STAGE_BYTES;
            for (uint32_t t = 0; t < 2u; ++t) {
                const uint32_t tile_row = (head * q_tile_rows + 2u * qb + t) * 4u;
                for (uint32_t k = 0; k < 8u; ++k) {
                    for (uint32_t qh = 0; qh < 2u; ++qh) {
                        noc_async_write(
                            stage + t * 8192u + k * 1024u + qh * WHS_TFACE,
                            o_gen.get_noc_addr(tile_row + k / 2u, (2u * qh + k % 2u) * WHS_TFACE),
                            WHS_TFACE);
                    }
                }
            }
            noc_async_write_barrier();
            line(scratch, WHS_S_O_WRITTEN) = ++o_next;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}

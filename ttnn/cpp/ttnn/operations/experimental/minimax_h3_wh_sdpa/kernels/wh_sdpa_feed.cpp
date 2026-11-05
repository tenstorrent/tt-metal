// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The WH SDPA op's K/V feed (wh_sdpa_op_common.h), on the data movement RISC that is not the I/O kernel's, of a group's
// reader cores (others return at once). Reader i takes blocks i, i + readers, ... of the group's stream: once the whole
// group has consumed the buffer's last use, it reads the block's 64 faces from DRAM into its own buffer (K's d slices
// [64 keys][16 d], then V^T's [64 d][16 keys] chunks from the in-place transposed V), multicasts the 32 KiB to the rest
// of the group and then the buffer's ready word, linked so the word lands after the data. Ported from the feed of
// tt-lab-private h3-wh src/wh_brisc.cpp (role_sdpa) to tt-metal's NoC API.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "wh_sdpa_op_common.h"

#ifndef WHS_FEED_DEPTH
#define WHS_FEED_DEPTH 4u
#endif

namespace {

inline volatile uint32_t& line(uint32_t scratch, uint32_t l) {
    return *reinterpret_cast<volatile uint32_t*>(whs_line(scratch, l));
}

}  // namespace

void kernel_main() {
    const uint32_t reader_index = get_arg_val<uint32_t>(WHS_A_READER_INDEX);
    if (reader_index == WHS_IDLE) {
        return;
    }
    const uint32_t scratch = get_arg_val<uint32_t>(WHS_A_SCRATCH);
    const uint32_t go = get_arg_val<uint32_t>(WHS_A_GO);
    const uint32_t k_tile_rows = get_arg_val<uint32_t>(WHS_A_K_TILE_ROWS);
    uint32_t blocks = 0;  // after GO, from the I/O kernel's valid key count
    const uint32_t bufs = get_arg_val<uint32_t>(WHS_A_BUFS);
    const uint32_t readers = get_arg_val<uint32_t>(WHS_A_READERS);
    const uint32_t group_cores = get_arg_val<uint32_t>(WHS_A_GROUP_CORES);
    const uint32_t rounds = get_arg_val<uint32_t>(WHS_A_ROUNDS);
    const uint32_t debug = get_arg_val<uint32_t>(WHS_A_DEBUG);
    const uint32_t x0 = get_arg_val<uint32_t>(WHS_A_GROUP_X0), y0 = get_arg_val<uint32_t>(WHS_A_GROUP_Y0);
    const uint32_t x1 = get_arg_val<uint32_t>(WHS_A_GROUP_X1), y1 = get_arg_val<uint32_t>(WHS_A_GROUP_Y1);
    const InterleavedAddrGen<true> k_gen = {.bank_base_address = get_arg_val<uint32_t>(WHS_A_K), .page_size = WHS_TILE};
    const InterleavedAddrGen<true> vt_gen = {
        .bank_base_address = get_arg_val<uint32_t>(WHS_A_VT), .page_size = WHS_TILE};
    const uint32_t kv = scratch + WHS_KV;
    uint32_t total = 0;
    const bool cast = group_cores > 1u && !(debug & WHS_DEBUG_NO_CASTS);
    const bool no_issue = debug & WHS_DEBUG_NO_ISSUE;
    const bool face1k = debug & WHS_FACE1K;
    const bool cast_by_io = (debug & WHS_CAST_BY_IO) && group_cores > 1u;
    const bool vt_full = debug & WHS_VT_FULL;
    uint32_t landed = 0;
    volatile uint32_t sink = 0;
    // The whole group rectangle, this reader included (loopback); NoC 1 multicasts take it from its high corner.
    const auto mcast = [&](uint32_t addr) -> uint64_t {
        return noc_index == 1 ? get_noc_multicast_addr(x1, y1, x0, y0, addr)
                              : get_noc_multicast_addr(x0, y0, x1, y1, addr);
    };

    // Block g's 64 faces into its buffer on transaction id `trid`: per tile one NoC state (the tile's bank and 512-byte
    // packets), then its 4 faces.
    const auto read_block = [&](uint32_t g, uint32_t trid) {
        const uint32_t r = g / blocks, j = g % blocks;
        const uint32_t head = get_arg_val<uint32_t>(WHS_A_ROUND0 + r) >> 16;
        const uint32_t buffer = kv + (g % bufs) * WHS_BLOCK;
        noc_async_read_set_trid(trid);
        for (uint32_t t = 0; t < 2u; ++t) {
            const uint32_t tile_row = (head * k_tile_rows + 2u * j + t) * 4u;
            for (uint32_t dc = 0; dc < 4u; ++dc) {
                // K: tile (2j + t, dc) face (rh, ch) is keys 32 t + 16 rh of d slice 2 dc + ch.
                const uint64_t k_tile = k_gen.get_noc_addr(tile_row + dc);
                if (no_issue) {
                    sink += uint32_t(k_tile) + uint32_t(k_tile >> 32);
                } else if (face1k) {
                    // Faces (0, ch) and (1, ch) adjacent: rows 32 t .. 32 t + 31 of d slice 2 dc + ch.
                    noc_async_read_one_packet_set_state(k_tile, 2u * WHS_TFACE);
                    for (uint32_t ch = 0; ch < 2u; ++ch) {
                        noc_async_read_one_packet_with_state_with_trid(
                            uint32_t(k_tile),
                            ch * 2u * WHS_TFACE,
                            buffer + (2u * dc + ch) * WHS_FACE + 2u * t * WHS_TFACE,
                            trid);
                    }
                } else {
                    noc_async_read_one_packet_set_state(k_tile, WHS_TFACE);
                    for (uint32_t fi = 0; fi < 4u; ++fi) {
                        const uint32_t rh = fi / 2u, ch = fi % 2u;
                        noc_async_read_one_packet_with_state_with_trid(
                            uint32_t(k_tile),
                            fi * WHS_TFACE,
                            buffer + (2u * dc + ch) * WHS_FACE + (2u * t + rh) * WHS_TFACE,
                            trid);
                    }
                }
                // V^T: transposed tile (2j + t, dc) face (rh, ch) is d 32 dc + 16 rh by keys 32 t + 16 ch, i.e. key
                // slice s = 2 t + ch, d half dc / 2, its rows 16 (2 (dc % 2) + rh).
                const uint64_t v_tile =
                    vt_gen.get_noc_addr(vt_full ? (head * 4u + dc) * k_tile_rows + 2u * j + t : tile_row + dc);
                if (no_issue) {
                    sink += uint32_t(v_tile) + uint32_t(v_tile >> 32);
                    continue;
                }
                if (face1k) {
                    // Faces (0, ch) and (1, ch): key slice 2 t + ch, rows 16 (2 (dc % 2)) .. + 31 of d half dc / 2.
                    noc_async_read_one_packet_set_state(v_tile, 2u * WHS_TFACE);
                    for (uint32_t ch = 0; ch < 2u; ++ch) {
                        const uint32_t s = 2u * t + ch, h = dc / 2u;
                        noc_async_read_one_packet_with_state_with_trid(
                            uint32_t(v_tile),
                            ch * 2u * WHS_TFACE,
                            buffer + (8u + 2u * s + h) * WHS_FACE + 2u * (dc % 2u) * WHS_TFACE,
                            trid);
                    }
                    continue;
                }
                noc_async_read_one_packet_set_state(v_tile, WHS_TFACE);
                for (uint32_t fi = 0; fi < 4u; ++fi) {
                    const uint32_t rh = fi / 2u, ch = fi % 2u;
                    const uint32_t s = 2u * t + ch, h = dc / 2u, i = 2u * (dc % 2u) + rh;
                    noc_async_read_one_packet_with_state_with_trid(
                        uint32_t(v_tile), fi * WHS_TFACE, buffer + (8u + 2u * s + h) * WHS_FACE + i * WHS_TFACE, trid);
                }
            }
        }
    };
    const auto cast_block = [&](uint32_t g, uint32_t trid) {
        const uint32_t b = g % bufs, use = g / bufs;
        const uint32_t buffer = kv + b * WHS_BLOCK;
        if (!(debug & (WHS_DEBUG_NO_READS | WHS_DEBUG_NO_ISSUE))) {
            noc_async_read_barrier_with_trid(trid);
        }
        if (cast_by_io) {
            line(scratch, WHS_S_LANDED) = ++landed;
            return;
        }
        volatile uint32_t& ready = line(scratch, WHS_S_READY + b);
        ready = use + 1u;
        if (group_cores > 1u) {
            if (cast) {
                if (!(debug & WHS_DEBUG_NO_PATH_RESERVE)) {
                    noc_async_write_multicast_loopback_src(buffer, mcast(buffer), WHS_BLOCK, group_cores, true);
                } else {
                    // The release's casts: no path reservation (see WHS_DEBUG_NO_PATH_RESERVE).
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
                }
            }
            const uint32_t ready_addr = whs_line(scratch, WHS_S_READY + b);
            noc_semaphore_set_multicast_loopback_src(ready_addr, mcast(ready_addr), group_cores);
        }
    };

    while (line(scratch, WHS_S_GO) != go) {
    }
    blocks = whs_key_blocks(line(scratch, WHS_S_VALID));
    total = rounds * blocks;
    const uint32_t start = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    // WHS_FEED_DEPTH blocks in flight: block n's reads go out before block n - (depth - 1) is cast, each block on
    // its own transaction id.
    constexpr uint32_t depth = WHS_FEED_DEPTH;
    uint32_t n = 0;
    // Ring gather: a head's blocks only once its K and V^T have landed on this chip.
    const uint32_t ring_counters = get_arg_val<uint32_t>(WHS_A_RING_COUNTERS);
    const uint32_t ring_target = get_arg_val<uint32_t>(WHS_A_RING_TARGET);
    const uint64_t ring_noc =
        ring_counters
            ? get_noc_addr(get_arg_val<uint32_t>(WHS_A_RING_X), get_arg_val<uint32_t>(WHS_A_RING_Y), ring_counters)
            : 0u;
    uint32_t landed_heads = 0;
    const auto wait_head = [&](uint32_t head) {
        if (landed_heads >> head & 1u) {
            return;
        }
        const uint32_t probe = whs_line(scratch, WHS_S_RING_PROBE);
        while (true) {
            noc_async_read(ring_noc + head * 16u, probe, 4u);
            noc_async_read_barrier();
            if (line(scratch, WHS_S_RING_PROBE) - ring_target < 0x80000000u) {
                break;
            }
            for (uint32_t i = 0; i < 64u; ++i) {
                __asm__ volatile("nop");
            }
        }
        landed_heads |= 1u << head;
    };
    for (uint32_t g = reader_index; g < total; g += readers, ++n) {
        const uint32_t b = g % bufs, use = g / bufs;
        while (line(scratch, WHS_S_CONSUMED + b) < group_cores * use) {
        }
        if (ring_counters) {
            wait_head(get_arg_val<uint32_t>(WHS_A_ROUND0 + g / blocks) >> 16);
        }
        if (!(debug & WHS_DEBUG_NO_READS)) {
            read_block(g, 1u + n % depth);
        }
        if (n + 1u >= depth) {
            const uint32_t m = n + 1u - depth;
            cast_block(reader_index + m * readers, 1u + m % depth);
        }
    }
    for (uint32_t m = n + 1u > depth ? n + 1u - depth : 0u; m < n; ++m) {
        cast_block(reader_index + m * readers, 1u + m % depth);
    }
    // Every buffer this reader fills consumed by the whole group: until then a cast may not have read its source, and
    // the group's posted increments may still land here.
    for (uint32_t b = reader_index; b < bufs && b < total; b += readers) {
        const uint32_t uses = (total - b + bufs - 1u) / bufs;
        while (line(scratch, WHS_S_CONSUMED + b) < group_cores * uses) {
        }
    }
    noc_async_write_barrier();
    line(scratch, WHS_S_STATS + 3u) = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - start;
}

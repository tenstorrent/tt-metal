// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker writer (BRISC / NoC1).
// 1. x arrival: for each round r, wait until its sender's multicast of round r has landed in
//    cb_x_resident, then push the round's tiles to compute. With two x senders round r comes on
//    channel r % 2 ("x_mcast" / "x_mcast_b", each its own Counter: local round r / 2); with one
//    sender "x_mcast_b" is attached absent and every round comes on "x_mcast".
// 2. store: write the block_m_tiles x block_n_tiles output block to the L1-interleaved y, one final
//    barrier. out_subblock_major: compute pushes one out_subblock_h x out_subblock_w subblock at a time
//    (SubblockMajor); each is written as soon as it lands. Else one TileRowMajor block.
//    y_store_noc_split: every other tile goes out on NoC0 (idle by then: the W reads are done), the
//    rest on this kernel's NoC1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
// Zones (permanent): x_wait_r0 (x round 0 arrival), x_wait_rest (rounds 1..), y_wait (waiting on compute's
// output; cumulative over subblocks), y_issue, y_barrier.
// Ablation: ABLATE_Y_STORE skips the y writes (waits, barrier and pop kept).
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_block = get_compile_time_arg_val(1);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(6);
    constexpr bool out_subblock_major = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);
    constexpr bool y_store_noc_split = get_compile_time_arg_val(10) != 0;
    constexpr auto y_args = TensorAccessorArgs<11>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();
    constexpr auto mc_b = McastArgs<
        get_named_compile_time_arg_val("x_mcast_b_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_b_rt_offset")>();
    constexpr uint32_t num_x_senders = mc_b.active ? 2 : 1;

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    constexpr uint32_t out_block_tiles = block_m_tiles * block_n_tiles;

#ifdef TAIL_L1ACC_FINAL
    // tail_store: compute's last K-step lands in cb_interm_acc (L1 accumulation, LastBlockTarget::Interm);
    // the writer reads it there. Compute's unpack thread pops cb_interm_acc for K-steps 0..n-2, so the
    // writer waits on the absolute received count (num_k_blocks-1 full blocks + the tiles it needs) and
    // never pops it.
    constexpr uint32_t cb_y_src = TAIL_CB_INTERM;
    volatile tt_reg_ptr uint32_t* interm_received = get_cb_tiles_received_ptr(cb_y_src);
    const uint16_t interm_received0 = (uint16_t)interm_received[0];
    auto wait_y = [&](uint32_t n) {
        const uint16_t target = (uint16_t)((num_k_blocks - 1) * out_block_tiles + n);
        while ((uint16_t)((uint16_t)interm_received[0] - interm_received0) < target) {
        }
    };
#else
    constexpr uint32_t cb_y_src = cb_out_block;
    auto wait_y = [&](uint32_t n) { cb_wait_front(cb_out_block, n); };
#endif
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    uint32_t cols[block_n_tiles];
    for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
        cols[nn] = get_arg_val<uint32_t>(1 + nn);
    }

    // ---- receive_x_rounds ----
    Noc noc;
    auto pipe = mc.receiver(noc);
    auto pipe_b = mc_b.optional_receiver(noc);
    auto receive_round = [&](uint32_t r) {
        cb_reserve_back(cb_x_resident, round_tiles);  // never blocks: x is fully resident
        if (num_x_senders == 1 || (r % num_x_senders) == 0) {
            pipe.receive(r / num_x_senders);
        } else {
            pipe_b->receive(r / num_x_senders);
        }
        cb_push_back(cb_x_resident, round_tiles);
    };
    {
        MaybeDeviceZoneScope("x_wait_r0");
        receive_round(0);
    }
    {
        MaybeDeviceZoneScope("x_wait_rest");
        for (uint32_t r = 1; r < num_k_blocks; ++r) {
            receive_round(r);
        }
    }

    // ---- store_block ----
    const uint32_t y_tile_bytes = get_tile_size(cb_out_block);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    uint32_t tile_i = 0;
    auto store_tile = [&](uint32_t m, uint32_t nn, uint32_t src) {
#ifdef ABLATE_Y_STORE
        return;
#endif
        const uint8_t noc_id = (y_store_noc_split && (tile_i++ & 1)) ? (1 - noc_index) : noc_index;
        noc_async_write(src, y_acc.get_noc_addr(m * tensor_n_tiles + cols[nn], 0, noc_id), y_tile_bytes, noc_id);
    };
#ifdef TAIL_PRECOMP_ADDR
    // tail_store: resolve every output tile's NoC address (L1-interleaved bank lookup = a division by the
    // L1 bank count) and NoC id up front (while waiting on compute), so the post-compute issue is one
    // noc_async_write per tile. Tile order = the order compute packs them (SubblockMajor or TileRowMajor).
    uint32_t page[out_block_tiles];
    {
        uint32_t i = 0;
        if constexpr (out_subblock_major) {
            for (uint32_t sm = 0; sm < block_m_tiles / out_subblock_h; ++sm) {
                for (uint32_t sn = 0; sn < block_n_tiles / out_subblock_w; ++sn) {
                    for (uint32_t h = 0; h < out_subblock_h; ++h) {
                        for (uint32_t w = 0; w < out_subblock_w; ++w) {
                            page[i++] = (sm * out_subblock_h + h) * tensor_n_tiles + cols[sn * out_subblock_w + w];
                        }
                    }
                }
            }
        } else {
            for (uint32_t m = 0; m < block_m_tiles; ++m) {
                for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
                    page[i++] = m * tensor_n_tiles + cols[nn];
                }
            }
        }
    }
    uint8_t dst_noc[out_block_tiles];
    for (uint32_t i = 0; i < out_block_tiles; ++i) {
        dst_noc[i] = (y_store_noc_split && (i & 1)) ? (1 - noc_index) : noc_index;
    }
#ifdef TAIL_NOC_BY_DIST
    // tail_store: pick each tile's NoC by hop count on the torus (NoC0 runs +x/+y, NoC1 -x/-y), in NoC0
    // coordinates. 1 = shorter path per tile (ties alternate); 2 = balanced: within each store group (one
    // subblock) the half of the tiles with the largest NoC0 advantage go on NoC0, the rest on NoC1.
    {
        constexpr uint32_t grp = out_subblock_major ? out_subblock_h * out_subblock_w : out_block_tiles;
        const int32_t W = noc_size_x, H = noc_size_y;
        const int32_t mx = my_x[0], my = my_y[0];
        int32_t adv[out_block_tiles];  // hops(NoC1) - hops(NoC0): > 0 -> NoC0 shorter
        for (uint32_t i = 0; i < out_block_tiles; ++i) {
            const uint64_t a = y_acc.get_noc_addr(page[i], 0, 0);
            const int32_t dx = (int32_t)NOC_UNICAST_ADDR_X(a) - mx, dy = (int32_t)NOC_UNICAST_ADDR_Y(a) - my;
            const int32_t hx0 = ((dx % W) + W) % W, hy0 = ((dy % H) + H) % H;
            const int32_t hx1 = (W - hx0) % W, hy1 = (H - hy0) % H;
            adv[i] = (hx1 + hy1) - (hx0 + hy0);
        }
        for (uint32_t g = 0; g < out_block_tiles; g += grp) {
            for (uint32_t i = g; i < g + grp; ++i) {
#if TAIL_NOC_BY_DIST == 2
                uint32_t rank = 0;
                for (uint32_t j = g; j < g + grp; ++j) {
                    rank += (adv[j] > adv[i]) || (adv[j] == adv[i] && j < i);
                }
                dst_noc[i] = (rank < (grp + 1) / 2) ? 0 : 1;
#elif TAIL_NOC_BY_DIST == 3
                // sanity reference: the LONGER path per tile (should be slower if the hop model is right)
                dst_noc[i] = adv[i] > 0 ? 1 : (adv[i] < 0 ? 0 : (uint8_t)((i - g) & 1));
#else
                dst_noc[i] = adv[i] > 0 ? 0 : (adv[i] < 0 ? 1 : (uint8_t)((i - g) & 1));
#endif
            }
        }
    }
#endif
    uint64_t dst_addr[out_block_tiles];
    for (uint32_t i = 0; i < out_block_tiles; ++i) {
        dst_addr[i] = y_acc.get_noc_addr(page[i], 0, dst_noc[i]);
    }
    {
        const uint32_t base = get_read_ptr(cb_y_src);
#ifdef TAIL_PER_TILE_WAIT
        // wait for each tile (compute pushes per tile with TAIL_PER_TILE_PUSH) and store it at once
        constexpr uint32_t step = 1;
#else
        constexpr uint32_t step = out_subblock_major ? out_subblock_h * out_subblock_w : out_block_tiles;
#endif
        auto issue = [&](uint32_t t0) {
            for (uint32_t t = t0; t < t0 + step; ++t) {
#ifndef ABLATE_Y_STORE
                noc_async_write(base + t * y_tile_bytes, dst_addr[t], y_tile_bytes, dst_noc[t]);
#endif
            }
        };
#ifdef TAIL_LAST_ZONE_ONLY
        // zones only on the last group (per-tile zones cost ~30 cycles each on the critical path)
        for (uint32_t t0 = 0; t0 + step < out_block_tiles; t0 += step) {
            wait_y(t0 + step);
            issue(t0);
        }
        {
            {
                MaybeDeviceZoneScope("y_wait");
                wait_y(out_block_tiles);
            }
            MaybeDeviceZoneScope("y_issue");
            issue(out_block_tiles - step);
        }
#else
        for (uint32_t t0 = 0; t0 < out_block_tiles; t0 += step) {
            {
                MaybeDeviceZoneScope("y_wait");
                wait_y(t0 + step);
            }
            MaybeDeviceZoneScope("y_issue");
            issue(t0);
        }
#endif
    }
#else
    if constexpr (out_subblock_major) {
        constexpr uint32_t sb_tiles = out_subblock_h * out_subblock_w;
        const uint32_t base = get_read_ptr(cb_out_block);
        uint32_t landed = 0;
        for (uint32_t sm = 0; sm < block_m_tiles / out_subblock_h; ++sm) {
            for (uint32_t sn = 0; sn < block_n_tiles / out_subblock_w; ++sn) {
                // Cumulative wait (nothing popped until the end): subblock `landed` sits at base + landed tiles.
                uint32_t src = base + landed * y_tile_bytes;
                landed += sb_tiles;
                {
                    MaybeDeviceZoneScope("y_wait");
                    cb_wait_front(cb_out_block, landed);
                }
                MaybeDeviceZoneScope("y_issue");
                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        store_tile(sm * out_subblock_h + h, sn * out_subblock_w + w, src);
                        src += y_tile_bytes;
                    }
                }
                // The tiles stay in the CB until the barrier below (one block, never reused): pop last.
            }
        }
    } else {
        {
            MaybeDeviceZoneScope("y_wait");
            cb_wait_front(cb_out_block, out_block_tiles);
        }
        MaybeDeviceZoneScope("y_issue");
        uint32_t src = get_read_ptr(cb_out_block);
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
                store_tile(m, nn, src);
                src += y_tile_bytes;
            }
        }
    }
#endif
    {
        MaybeDeviceZoneScope("y_barrier");
#ifdef TAIL_FLUSH_ONLY
        // REFERENCE ONLY (not completion-safe): wait for the writes to leave this core, not for their acks.
        noc_async_writes_flushed();
        if constexpr (y_store_noc_split) {
            noc_async_writes_flushed(1 - noc_index);
        }
#else
        noc_async_write_barrier();
        if constexpr (y_store_noc_split) {
            noc_async_write_barrier(1 - noc_index);
        }
#endif
    }
#ifndef TAIL_L1ACC_FINAL
    cb_pop_front(cb_out_block, out_block_tiles);
#endif
}

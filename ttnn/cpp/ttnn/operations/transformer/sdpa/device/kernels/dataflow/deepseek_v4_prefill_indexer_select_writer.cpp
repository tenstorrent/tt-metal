// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_indexer_select writer. A split > 0 that scanned a block copies its finished top-k survivors
// (fp32 value tile and uint32 index tile per query) to the DRAM scratch tensor; every split > 0 then signals its
// leader. The leader (split 0) waits for every peer, feeds the survivors of the peers that scanned a block to its
// compute in merge order, and writes the merged output rows.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"

namespace dis = deepseek_v4_prefill_indexer_select;

void kernel_main() {
    using namespace tt::constants;
    constexpr uint32_t num_tokens = get_compile_time_arg_val(dis::writer_ct_arg::NUM_TOKENS);
    constexpr uint32_t num_splits = get_compile_time_arg_val(dis::writer_ct_arg::NUM_SPLITS);
    constexpr uint32_t cb_surv_v = get_compile_time_arg_val(dis::writer_ct_arg::CB_SURV_V);
    constexpr uint32_t cb_surv_i = get_compile_time_arg_val(dis::writer_ct_arg::CB_SURV_I);
    constexpr uint32_t cb_peer = get_compile_time_arg_val(dis::writer_ct_arg::CB_PEER);
    constexpr uint32_t cb_out = get_compile_time_arg_val(dis::writer_ct_arg::CB_OUT);
    constexpr uint32_t cb_done = get_compile_time_arg_val(dis::writer_ct_arg::CB_DONE);
    constexpr uint32_t cb_plan = get_compile_time_arg_val(dis::writer_ct_arg::CB_PLAN);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(dis::writer_ct_arg::SEMAPHORE);

    constexpr auto out_args = TensorAccessorArgs<dis::writer_ct_arg::END, dis::writer_common_arg::END>();
    constexpr auto scratch_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(dis::writer_common_arg::OUTPUT_ADDRESS));
    const auto scratch =
        TensorAccessor(scratch_args, get_common_arg_val<uint32_t>(dis::writer_common_arg::SCRATCH_ADDRESS));

    const uint32_t first_q_tile = get_arg_val<uint32_t>(dis::core_arg::FIRST_Q_TILE);
    const uint32_t q_tile_stride = get_arg_val<uint32_t>(dis::core_arg::Q_TILE_STRIDE);
    const uint32_t num_q_tiles = get_arg_val<uint32_t>(dis::core_arg::NUM_Q_TILES);
    const uint32_t split = get_arg_val<uint32_t>(dis::core_arg::SPLIT);
    const uint32_t leader_x = get_arg_val<uint32_t>(dis::core_arg::LEADER_X);
    const uint32_t leader_y = get_arg_val<uint32_t>(dis::core_arg::LEADER_Y);

    constexpr uint32_t surv_tile_bytes = TILE_HW * sizeof(uint32_t);
    constexpr uint32_t out_row_bytes = dis::TOPK * sizeof(uint32_t);
    // Scratch page (peer p, query q, value / index) = (p * T + q) * 2 + {0, 1}.
    const auto scratch_page = [](uint32_t peer, uint32_t token) { return (peer * num_tokens + token) * 2; };

    Noc noc;
    experimental::CB surv_v_cb(cb_surv_v), surv_i_cb(cb_surv_i), peer_cb(cb_peer), out_cb(cb_out), done_cb(cb_done),
        plan_cb(cb_plan);
    const uint32_t semaphore_addr = get_semaphore(semaphore_id);

    for (uint32_t unit = 0; unit < num_q_tiles; ++unit) {
        const uint32_t token0 = (first_q_tile + unit * q_tile_stride) * TILE_HEIGHT;
        plan_cb.wait_front(1);
        const uint32_t tile_blocks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_cb.get_read_ptr());
        plan_cb.pop_front(1);
        const auto scanned = [&](uint32_t s) {
            return dis::split_first_block(s + 1, tile_blocks, num_splits) >
                   dis::split_first_block(s, tile_blocks, num_splits);
        };
        if (split != 0 && !scanned(split)) {
            noc_semaphore_inc(get_noc_addr(leader_x, leader_y, semaphore_addr), 1);
            continue;
        }
        if (split != 0) {
            // Compute packs survivors at fixed slots and never pushes them; the token says they are final.
            done_cb.wait_front(1);
            const uint32_t v_base = surv_v_cb.get_read_ptr();
            const uint32_t i_base = surv_i_cb.get_read_ptr();
            for (uint32_t s = 0; s < dis::ROWS; ++s) {
                const uint32_t page = scratch_page(split - 1, token0 + s);
                noc.async_write(
                    CoreLocalMem<uint32_t>(v_base + s * surv_tile_bytes),
                    scratch,
                    surv_tile_bytes,
                    {},
                    {.page_id = page});
                noc.async_write(
                    CoreLocalMem<uint32_t>(i_base + s * surv_tile_bytes),
                    scratch,
                    surv_tile_bytes,
                    {},
                    {.page_id = page + 1});
            }
            noc.async_write_barrier();
            noc_semaphore_inc(get_noc_addr(leader_x, leader_y, semaphore_addr), 1);
            done_cb.pop_front(1);
            continue;
        }

        if constexpr (num_splits > 1) {
            auto* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
            noc_semaphore_wait(semaphore, num_splits - 1);
            noc_semaphore_set(semaphore, 0);
        }
        for (uint32_t s = 0; s < dis::ROWS; ++s) {
            for (uint32_t peer = 0; peer + 1 < num_splits; ++peer) {
                if (!scanned(peer + 1)) {
                    continue;
                }
                const uint32_t page = scratch_page(peer, token0 + s);
                peer_cb.reserve_back(2);
                const uint32_t ptr = peer_cb.get_write_ptr();
                noc.async_read(scratch, CoreLocalMem<uint32_t>(ptr), surv_tile_bytes, {.page_id = page}, {});
                noc.async_read(
                    scratch, CoreLocalMem<uint32_t>(ptr + surv_tile_bytes), surv_tile_bytes, {.page_id = page + 1}, {});
                noc.async_read_barrier();
                peer_cb.push_back(2);
            }
            out_cb.wait_front(1);
            noc.async_write(
                CoreLocalMem<uint32_t>(out_cb.get_read_ptr()), out, out_row_bytes, {}, {.page_id = token0 + s});
            noc.async_writes_flushed();
            out_cb.pop_front(1);
        }
        noc.async_write_barrier();
    }
    noc_async_atomic_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_indexer_select writer. A split > 0 that scanned a block copies its finished top-k survivors
// (fp32 value tile and uint32 index tile per query) to the DRAM scratch tensor; every split > 0 then signals its
// leader. The leader (split 0) waits for every peer, feeds the survivors of the peers that scanned a block to its
// compute in merge order, and writes the merged output rows.
//
// Candidate mode (instead of the entry selection): per scored group, this kernel reduces each query's score rows
// to candidate-block maxima (the block holding the query's newest entry pinned to +inf) into the staged chunk
// compute sorts; each split keeps its per-query top-CAND_TOPK block survivors in a second DRAM scratch between
// chunks, and the leader merges them all and writes each query's candidate block ids.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"

namespace dis = deepseek_v4_prefill_indexer_select;

namespace {

// Order key of a bf16: unsigned order matches the float order.
inline uint32_t bf16_order(uint32_t bits) { return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u); }

// The best of CAND_BLOCK consecutive bf16 scores (4-byte aligned).
inline uint16_t block_max(volatile tt_l1_ptr uint32_t* words) {
    uint32_t best = 0;
    uint16_t value = dis::BF16_NEG_INF;
    for (uint32_t i = 0; i < dis::CAND_BLOCK / 2; ++i) {
        const uint32_t pair = words[i];
        const uint32_t lo = pair & 0xFFFFu;
        const uint32_t hi = pair >> 16;
        const uint32_t klo = bf16_order(lo);
        const uint32_t khi = bf16_order(hi);
        if (klo > best) {
            best = klo;
            value = lo;
        }
        if (khi > best) {
            best = khi;
            value = hi;
        }
    }
    return value;
}

}  // namespace

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
    constexpr bool candidates = get_compile_time_arg_val(dis::writer_ct_arg::CANDIDATES) != 0;
    constexpr uint32_t cb_scores = get_compile_time_arg_val(dis::writer_ct_arg::CB_SCORES);
    constexpr uint32_t cb_group = get_compile_time_arg_val(dis::writer_ct_arg::CB_GROUP);
    constexpr uint32_t cb_group_ack = get_compile_time_arg_val(dis::writer_ct_arg::CB_GROUP_ACK);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(dis::writer_ct_arg::CB_STAGE);
    constexpr uint32_t cb_csurv_in = get_compile_time_arg_val(dis::writer_ct_arg::CB_CSURV_IN);
    constexpr uint32_t cb_csurv_out = get_compile_time_arg_val(dis::writer_ct_arg::CB_CSURV_OUT);
    constexpr uint32_t cb_cand_out = get_compile_time_arg_val(dis::writer_ct_arg::CB_CAND_OUT);

    constexpr auto out_args = TensorAccessorArgs<dis::writer_ct_arg::END, dis::writer_common_arg::END>();
    constexpr auto scratch_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto cand_out_args = TensorAccessorArgs<
        scratch_args.next_compile_time_args_offset(),
        scratch_args.next_common_runtime_args_offset()>();
    constexpr auto cand_scratch_args = TensorAccessorArgs<
        cand_out_args.next_compile_time_args_offset(),
        cand_out_args.next_common_runtime_args_offset()>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(dis::writer_common_arg::OUTPUT_ADDRESS));
    const auto scratch =
        TensorAccessor(scratch_args, get_common_arg_val<uint32_t>(dis::writer_common_arg::SCRATCH_ADDRESS));
    const auto cand_out =
        TensorAccessor(cand_out_args, get_common_arg_val<uint32_t>(dis::writer_common_arg::CAND_OUTPUT_ADDRESS));
    const auto cand_scratch =
        TensorAccessor(cand_scratch_args, get_common_arg_val<uint32_t>(dis::writer_common_arg::CAND_SCRATCH_ADDRESS));

    const uint32_t first_q_tile = get_arg_val<uint32_t>(dis::core_arg::FIRST_Q_TILE);
    const uint32_t q_tile_stride = get_arg_val<uint32_t>(dis::core_arg::Q_TILE_STRIDE);
    const uint32_t num_q_tiles = get_arg_val<uint32_t>(dis::core_arg::NUM_Q_TILES);
    const uint32_t split = get_arg_val<uint32_t>(dis::core_arg::SPLIT);
    const uint32_t leader_x = get_arg_val<uint32_t>(dis::core_arg::LEADER_X);
    const uint32_t leader_y = get_arg_val<uint32_t>(dis::core_arg::LEADER_Y);

    constexpr uint32_t surv_tile_bytes = TILE_HW * sizeof(uint32_t);
    constexpr uint32_t score_tile_bytes = TILE_HW * sizeof(uint16_t);
    constexpr uint32_t out_row_bytes = dis::TOPK * sizeof(uint32_t);
    constexpr uint32_t cand_out_row_bytes = dis::CAND_TOPK * sizeof(uint32_t);
    constexpr uint32_t stage_tiles = dis::ROWS * dis::CAND_ROW_BYTES / score_tile_bytes;
    // Scratch page (peer p, query q, value / index) = (p * T + q) * 2 + {0, 1}.
    const auto scratch_page = [](uint32_t peer, uint32_t token) { return (peer * num_tokens + token) * 2; };
    // Candidate scratch page (split, query, survivor tile) = (split * T + q) * CAND_SURV_TILES + tile.
    const auto cand_page = [](uint32_t owner, uint32_t token) {
        return (owner * num_tokens + token) * dis::CAND_SURV_TILES;
    };

    Noc noc;
    experimental::CB surv_v_cb(cb_surv_v), surv_i_cb(cb_surv_i), peer_cb(cb_peer), out_cb(cb_out), done_cb(cb_done),
        plan_cb(cb_plan);
    experimental::CB scores_cb(cb_scores), group_cb(cb_group), ack_cb(cb_group_ack), stage_cb(cb_stage),
        csurv_in_cb(cb_csurv_in), csurv_out_cb(cb_csurv_out), cand_out_cb(cb_cand_out);
    const uint32_t semaphore_addr = get_semaphore(semaphore_id);

    // One survivor between this kernel and compute, through candidate scratch page `page`.
    const auto read_survivor = [&](uint32_t page) {
        csurv_in_cb.reserve_back(dis::CAND_SURV_TILES);
        const uint32_t ptr = csurv_in_cb.get_write_ptr();
        for (uint32_t t = 0; t < dis::CAND_SURV_TILES; ++t) {
            noc.async_read(
                cand_scratch,
                CoreLocalMem<uint32_t>(ptr + t * surv_tile_bytes),
                surv_tile_bytes,
                {.page_id = page + t},
                {});
        }
        noc.async_read_barrier();
        csurv_in_cb.push_back(dis::CAND_SURV_TILES);
    };
    const auto write_survivor = [&](uint32_t page) {
        csurv_out_cb.wait_front(dis::CAND_SURV_TILES);
        const uint32_t ptr = csurv_out_cb.get_read_ptr();
        for (uint32_t t = 0; t < dis::CAND_SURV_TILES; ++t) {
            noc.async_write(
                CoreLocalMem<uint32_t>(ptr + t * surv_tile_bytes),
                cand_scratch,
                surv_tile_bytes,
                {},
                {.page_id = page + t});
        }
        // A later chunk reads this survivor back.
        noc.async_write_barrier();
        csurv_out_cb.pop_front(dis::CAND_SURV_TILES);
    };

    for (uint32_t unit = 0; unit < num_q_tiles; ++unit) {
        const uint32_t token0 = (first_q_tile + unit * q_tile_stride) * TILE_HEIGHT;
        plan_cb.wait_front(1);
        uint32_t limits[dis::ROWS];
        uint32_t num_blocks, first_entry, tile_blocks;
        {
            auto* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_cb.get_read_ptr());
            num_blocks = plan[dis::meta_word::NUM_BLOCKS];
            first_entry = plan[dis::meta_word::FIRST_ENTRY];
            tile_blocks = plan[dis::meta_word::TILE_BLOCKS];
            for (uint32_t s = 0; s < dis::ROWS; ++s) {
                limits[s] = plan[dis::meta_word::LIMITS + s];
            }
        }
        plan_cb.pop_front(1);
        const auto scanned = [&](uint32_t s) {
            return dis::split_first_block(s + 1, tile_blocks, num_splits) >
                   dis::split_first_block(s, tile_blocks, num_splits);
        };

        if constexpr (candidates) {
            // Compute packs every group at the base of cb_scores (it never wraps), so this kernel, which never
            // pops it, reads a group's score rows at fixed offsets from its own read pointer.
            const uint32_t scores_base = scores_cb.get_read_ptr();
            const uint32_t first_block = first_entry / dis::TOPK;
            const uint32_t end_block = first_block + num_blocks;
            bool seeded[dis::ROWS];
            for (uint32_t s = 0; s < dis::ROWS; ++s) {
                seeded[s] = false;
            }
            uint32_t stage_base = 0;
            for (uint32_t group = first_block; group < end_block;) {
                const uint32_t group_end_aligned = (group / dis::GROUP_BLOCKS + 1) * dis::GROUP_BLOCKS;
                const uint32_t group_end = group_end_aligned < end_block ? group_end_aligned : end_block;
                const uint32_t chunk = group / dis::CAND_CHUNK_BLOCKS;
                const uint32_t chunk_block0 = chunk * dis::CAND_CHUNK_BLOCKS;
                if (group == first_block || group == chunk_block0) {
                    stage_cb.reserve_back(stage_tiles);
                    stage_base = stage_cb.get_write_ptr();
                    // Staged positions before the split's first block are not its blocks: -inf.
                    const uint32_t head = (group - chunk_block0) * dis::CAND_PER_BLOCK;
                    for (uint32_t s = 0; s < dis::ROWS && head > 0; ++s) {
                        auto* row =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_base + s * dis::CAND_ROW_BYTES);
                        for (uint32_t i = 0; i < head / 2; ++i) {
                            row[i] = (uint32_t{dis::BF16_NEG_INF} << 16) | dis::BF16_NEG_INF;
                        }
                    }
                }
                group_cb.wait_front(1);
                for (uint32_t s = 0; s < dis::ROWS; ++s) {
                    const uint32_t seen_blocks = (limits[s] + dis::TOPK - 1) / dis::TOPK;
                    const uint32_t row_end = seen_blocks < group_end ? seen_blocks : group_end;
                    const uint32_t visible = (limits[s] + dis::CAND_BLOCK - 1) / dis::CAND_BLOCK;
                    for (uint32_t b = group; b < row_end; ++b) {
                        auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            scores_base + ((b - group) * dis::ROWS + s) * score_tile_bytes);
                        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                            stage_base + s * dis::CAND_ROW_BYTES + (b - chunk_block0) * dis::CAND_PER_BLOCK * 2);
                        for (uint32_t k = 0; k < dis::CAND_PER_BLOCK; ++k) {
                            const uint32_t cblock = b * dis::CAND_PER_BLOCK + k;
                            if (cblock >= visible) {
                                break;
                            }
                            // The newest block is pinned (it is also the only one a query sees part of).
                            dst[k] =
                                cblock + 1 == visible ? dis::BF16_POS_INF : block_max(src + k * dis::CAND_BLOCK / 2);
                        }
                    }
                }
                group_cb.pop_front(1);
                ack_cb.reserve_back(1);
                ack_cb.push_back(1);
                if (group_end % dis::CAND_CHUNK_BLOCKS == 0 || group_end == end_block) {
                    stage_cb.push_back(stage_tiles);
                    for (uint32_t s = 0; s < dis::ROWS; ++s) {
                        if (dis::cand_active(limits[s], chunk, first_block, end_block) == 0) {
                            continue;
                        }
                        const uint32_t page = cand_page(split, token0 + s);
                        if (seeded[s]) {
                            read_survivor(page);
                        }
                        write_survivor(page);
                        seeded[s] = true;
                    }
                }
                group = group_end;
            }
            // Every row of a split the leader merges has a survivor, an all -inf one if it saw nothing here.
            if (split == 0 || num_blocks > 0) {
                for (uint32_t s = 0; s < dis::ROWS; ++s) {
                    if (!seeded[s]) {
                        write_survivor(cand_page(split, token0 + s));
                    }
                }
            }
            if (split != 0) {
                noc_semaphore_inc(get_noc_addr(leader_x, leader_y, semaphore_addr), 1);
                continue;
            }
            if constexpr (num_splits > 1) {
                auto* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
                noc_semaphore_wait(semaphore, num_splits - 1);
                noc_semaphore_set(semaphore, 0);
            }
            for (uint32_t s = 0; s < dis::ROWS; ++s) {
                read_survivor(cand_page(0, token0 + s));
                for (uint32_t peer = 1; peer < num_splits; ++peer) {
                    if (scanned(peer)) {
                        read_survivor(cand_page(peer, token0 + s));
                    }
                }
                cand_out_cb.wait_front(2);
                noc.async_write(
                    CoreLocalMem<uint32_t>(cand_out_cb.get_read_ptr()),
                    cand_out,
                    cand_out_row_bytes,
                    {},
                    {.page_id = token0 + s});
                noc.async_writes_flushed();
                cand_out_cb.pop_front(2);
            }
            noc.async_write_barrier();
            continue;
        }

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

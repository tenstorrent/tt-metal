// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader of paged_scaled_dot_product_attention_decode. Thread plan: R == C, so reader hart n feeds NEO n alone on
// every STRIDED ring (the per-NEO streams differ in length: each worker has its own range of the virtual sequence).
//
// Per user of this cluster (op_design.md, Block schedule):
//   load_position_block  cur_pos page and the user's page-table row into this hart's scratchpad region (tagged,
//                        one barrier); the plan from decode_plan.hpp; the raw position to compute (cb_cur_pos).
//   load_q_block         the user's tensor_d_tiles query tiles, through q's TensorAccessor (sharded or interleaved).
//   per head segment     load_mask_block: body mask (rows outside the head's group masked) and last mask (also the
//                        columns after cur_pos), built in L1 by this hart.
//   per chunk            load_k_block: K tiles in (t, j) order; load_v_block: V tiles in (d-sub-block, t, j) order.
//   combiner (rank 0)    wait for every partial (this hart's semaphore), then each partial's [m, O share, l].
//
// Ring protocol: explicit sync on every ring (disable_dfb_implicit_sync_for_all), every transfer tagged with user
// transaction id 0. Implicit sync is not usable here: its DM0 ISR credits a ring in batches that take entries from
// every producer hart (DataflowBuffer.md, "How the ISR threshold is programmed"), and the harts' streams differ in
// length (each feeds one worker's range), so a hart that finished its range would hold back the other NEOs' last
// tiles until kernel exit, while it waits on the combine: deadlock. Reads go in blocks (one tile-row of K, one
// d-sub-block of V, all of q), one barrier per block.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

#include "decode_plan.hpp"

namespace {

constexpr uint32_t FACE_DIM = 16;

// Element offset of (row, col) in a 32x32 tile stored as four 16x16 faces.
inline uint32_t face_offset(uint32_t row, uint32_t col) {
    return ((row / FACE_DIM) * 2 + col / FACE_DIM) * (FACE_DIM * FACE_DIM) + (row % FACE_DIM) * FACE_DIM +
           col % FACE_DIM;
}

// Write NEG_BIG into columns [col_begin, 32) of `row` (the pattern lanes only; the tile was zeroed by the DM engine).
template <bool fp32>
inline void mask_row(uint32_t tile_addr, uint32_t row, uint32_t col_begin) {
    if constexpr (fp32) {
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(tile_addr);
        for (uint32_t col = col_begin; col < decode_plan::TILE_DIM; ++col) {
            p[face_offset(row, col)] = decode_plan::NEG_BIG_FP32_BITS;
        }
    } else {
        volatile uint16_t* p = reinterpret_cast<volatile uint16_t*>(tile_addr);
        for (uint32_t col = col_begin; col < decode_plan::TILE_DIM; ++col) {
            p[face_offset(row, col)] = static_cast<uint16_t>(decode_plan::NEG_BIG_FP32_BITS >> 16);
        }
    }
}

// One explicit-sync block: reserve n entries, one tagged read per entry (entries of this hart's slice are
// stride_size apart), one barrier, push. Every block size divides its ring's per-NEO capacity and every block
// starts at a multiple of it, so a block never wraps the ring.
template <typename Accessor, typename Ring, typename PageFn>
inline void read_block(const Noc& noc, const Accessor& acc, Ring& ring, uint32_t n, PageFn&& page_of) {
    constexpr NocOptVals user_txn = {.trid = 0};
    ring.reserve_back(n);
    const uint32_t bytes = ring.get_entry_size();
    const uint32_t stride = ring.get_stride_size();
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read<NocOptions::TXN_ID>(
            acc, ring, bytes, {.page_id = page_of(i)}, {.offset_bytes = i * stride}, user_txn);
    }
    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
    ring.push_back(n);
}

}  // namespace

void kernel_main() {
    const uint32_t user_start = get_arg(args::user_start);
    const uint32_t user_count = get_arg(args::user_count);
    const uint32_t cluster_rank = get_arg(args::cluster_rank);

    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t num_heads = get_arg(args::num_heads);
    constexpr uint32_t kv_heads = get_arg(args::kv_heads);
    constexpr uint32_t Dt = get_arg(args::tensor_d_tiles);
    constexpr uint32_t d_share_tiles = get_arg(args::d_share_tiles);
    constexpr uint32_t block_pos_tiles = get_arg(args::block_pos_tiles);
    constexpr uint32_t pages_per_user = get_arg(args::pages_per_user);
    constexpr uint32_t tensor_pos_tiles = get_arg(args::tensor_pos_tiles);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t clusters_per_user = get_arg(args::clusters_per_user);
    constexpr uint32_t max_workers_per_user = get_arg(args::max_workers_per_user);
    constexpr uint32_t k_chunk_tiles = get_arg(args::k_chunk_tiles);
    constexpr uint32_t merge_cost = get_arg(args::merge_cost_units);
    constexpr uint32_t has_row_groups = get_arg(args::has_row_groups);
    constexpr uint32_t pv_subblock_tiles = get_arg(args::pv_subblock_tiles);
    constexpr bool inter_fp32 = get_arg(args::inter_fp32) != 0;
    constexpr uint32_t pos_page_bytes = get_arg(args::pos_page_bytes);
    constexpr uint32_t table_page_bytes = get_arg(args::table_page_bytes);
    constexpr uint32_t hart_scratch = get_arg(args::reader_hart_scratch);
    constexpr uint32_t group_size = num_heads / kv_heads;
    constexpr uint32_t partial_tiles = Dt + 2;
    static_assert(get_arg(args::reader_threads) == C, "each reader hart feeds exactly one NEO");

    const uint32_t hart = get_my_thread_id();  // == the NEO this hart feeds

    DataflowBuffer cur_pos_ring(dfb::cur_pos);
    DataflowBuffer scaler_ring(dfb::reduce_scaler);
    DataflowBuffer q_ring(dfb::q_tiles);
    DataflowBuffer k_ring(dfb::k_tiles);
    DataflowBuffer v_ring(dfb::v_tiles);
    DataflowBuffer mask_ring(dfb::score_masks);
    DataflowBuffer partials_ring(dfb::partials_in);
    Noc noc;
    const auto acc_q = TensorAccessor(tensor::q);
    const auto acc_k = TensorAccessor(tensor::k);
    const auto acc_v = TensorAccessor(tensor::v);
    const auto acc_table = TensorAccessor(tensor::page_table);
    const auto acc_pos = TensorAccessor(tensor::cur_pos);
    const auto acc_partials = TensorAccessor(tensor::partials);
    Scratchpad<uint32_t> scratch(scratch::scratch);
    const SemaphoreBindingToken sem_tokens[] = {
        sem::partials_ready_0, sem::partials_ready_1, sem::partials_ready_2, sem::partials_ready_3};
    Semaphore<> partials_ready(sem_tokens[hart]);
    constexpr NocOptVals user_txn = {.trid = 0};

    // This hart's scratchpad region: [positions page][page-table row page], each 64 B aligned.
    constexpr uint32_t pos_region = 0;
    constexpr uint32_t table_region = (pos_page_bytes + 63) / 64 * 64;
    const uint32_t region = hart * hart_scratch;

    // The reduce scaler (1.0 serves both the MAX and the SUM row reduce), one per NEO.
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_ring, 1.0f);

    uint32_t partials_expected = 0;
    for (uint32_t u = user_start; u < user_start + user_count; ++u) {
        // ---- load_position_block: the positions page and the user's page-table row, one barrier.
        noc.async_read<NocOptions::TXN_ID>(
            acc_pos, scratch, pos_page_bytes, {.page_id = 0}, {.offset_bytes = region + pos_region}, user_txn);
        noc.async_read<NocOptions::TXN_ID>(
            acc_table, scratch, table_page_bytes, {.page_id = u}, {.offset_bytes = region + table_region}, user_txn);
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        // Invalidate this hart's region once: the NoC wrote it behind the DM caches. Held for the user.
        auto region_lock = scratch.scoped_lock(region / 4, hart_scratch / 4);
        const uint32_t raw_pos = scratch[(region + pos_region) / 4 + u];
        const auto plan = decode_plan::make_plan<tensor_pos_tiles, kv_heads, C, clusters_per_user, merge_cost>(
            raw_pos, cluster_rank, hart);

        // The raw position to this hart's NEO: compute derives the same plan from it.
        cur_pos_ring.reserve_back(1);
        {
            const uint32_t addr = cur_pos_ring.get_write_ptr();
            reinterpret_cast<volatile uint32_t*>(addr)[0] = raw_pos;
            flush_l2_cache_range(addr, 64);
        }
        cur_pos_ring.push_back(1);
        if (!plan.cluster_active) {
            continue;  // this cluster takes no part in the user's work
        }

        if (plan.unit_count > 0) {
            // ---- load_q_block
            read_block(noc, acc_q, q_ring, Dt, [&](uint32_t j) { return u * Dt + j; });

            auto block_of = [&](uint32_t t) __attribute__((always_inline)) {
                const int32_t sb = static_cast<int32_t>(scratch[(region + table_region) / 4 + t / block_pos_tiles]);
                return sb < 0 ? 0u : (static_cast<uint32_t>(sb) >= num_blocks ? num_blocks - 1 : static_cast<uint32_t>(sb));
            };
            auto kv_page = [&](uint32_t block, uint32_t head, uint32_t t, uint32_t j) __attribute__((always_inline)) {
                return ((block * kv_heads + head) * block_pos_tiles + t % block_pos_tiles) * Dt + j;
            };

            decode_plan::for_each_chunk<k_chunk_tiles>(
                plan,
                // ---- load_mask_block: [body, last] for this head segment.
                [&](uint32_t head, uint32_t, uint32_t) {
                    for (uint32_t m = 0; m < 2; ++m) {
                        const bool last = (m == 1);
                        mask_ring.reserve_back(1);
                        const uint32_t addr = mask_ring.get_write_ptr();
                        const uint32_t bytes = mask_ring.get_tile_size();
                        noc.async_write_zeros(mask_ring, bytes);
                        noc.write_zeros_l1_barrier();
                        for (uint32_t row = 0; row < decode_plan::TILE_DIM; ++row) {
                            const bool in_group =
                                !has_row_groups || (row < num_heads && row / group_size == head);
                            if (!in_group) {
                                mask_row<inter_fp32>(addr, row, 0);
                            } else if (last && plan.last_col + 1 < decode_plan::TILE_DIM) {
                                mask_row<inter_fp32>(addr, row, plan.last_col + 1);
                            }
                        }
                        flush_l2_cache_range(addr, bytes);
                        mask_ring.push_back(1);
                    }
                },
                // ---- load_k_block / load_v_block
                [&](uint32_t head, uint32_t t0, uint32_t n, bool) {
                    for (uint32_t t = t0; t < t0 + n; ++t) {
                        const uint32_t block = block_of(t);
                        read_block(noc, acc_k, k_ring, Dt, [&](uint32_t j) { return kv_page(block, head, t, j); });
                    }
                    for (uint32_t b0 = 0; b0 < Dt; b0 += pv_subblock_tiles) {
                        for (uint32_t t = t0; t < t0 + n; ++t) {
                            const uint32_t block = block_of(t);
                            read_block(noc, acc_v, v_ring, pv_subblock_tiles, [&](uint32_t jj) {
                                return kv_page(block, head, t, b0 + jj);
                            });
                        }
                    }
                },
                []() {});
        }

        // ---- combiner: wait for every partial of the user, then stream [m, l, O share] of each.
        if (cluster_rank == 0) {
            // Cumulative count: each partial of every user of this cluster adds one; the semaphore is set back to 0
            // once all arrived.
            partials_expected += plan.workers_active;
            partials_ready.wait_min(partials_expected);
            for (uint32_t k = 0; k < plan.workers_active; ++k) {
                const uint32_t slot_page = (u * max_workers_per_user + k) * partial_tiles;
                // [m, O share, l]: the order compute consumes them (stats, then beta * [O share, l]).
                read_block(noc, acc_partials, partials_ring, 1, [&](uint32_t) { return slot_page + Dt; });
                read_block(noc, acc_partials, partials_ring, d_share_tiles, [&](uint32_t i) {
                    const uint32_t d = i * C + hart;
                    return slot_page + (d < Dt ? d : i * C);  // short last round: NEO re-merges d-tile i*C
                });
                read_block(noc, acc_partials, partials_ring, 1, [&](uint32_t) { return slot_page + Dt + 1; });
            }
        }
    }
    if (partials_expected > 0) {
        partials_ready.set(0);  // every increment of this launch has arrived: leave it at 0 for the next one
    }
    // Hand the hart back with the user id in its tag register: a program that runs next with untagged transfers
    // would otherwise inherit a ring's id (target page, ring protocols).
    noc_async_read_set_trid(0, noc.get_noc_id());
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer of paged_scaled_dot_product_attention_decode. Writer hart h of W drains NEOs h, h+W, ... (C/W of them) in
// strict rotation on both STRIDED rings: a whole partial per NEO on the partials ring (one barrier each), one entry
// per NEO on the output ring. Every NEO pushes the same number of entries per user (tensor_d_tiles + 2 partial
// tiles; d_share_tiles output tiles, a short last round re-stored by NEO).
//
// Per user of this cluster:
//   store_partial_block  each served NEO's partial [O tiles, m, l] into its worker slot of the partials scratch;
//                        once all are written (barriered): one increment of every combiner hart's "partials ready"
//                        semaphore per partial (C semaphores: reader hart n of the combiner waits on number n).
//   store_output_block   (combiner, rank 0) each served NEO's share of the head dimension.
//
// Explicit sync on both rings (disable_dfb_implicit_sync_for_all): every transfer carries user transaction id 0.

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

#include "decode_plan.hpp"

void kernel_main() {
    const uint32_t user_start = get_arg(args::user_start);
    const uint32_t user_count = get_arg(args::user_count);
    const uint32_t cluster_rank = get_arg(args::cluster_rank);
    const uint32_t combiner_x = get_arg(args::combiner_x);
    const uint32_t combiner_y = get_arg(args::combiner_y);

    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t kv_heads = get_arg(args::kv_heads);
    constexpr uint32_t Dt = get_arg(args::tensor_d_tiles);
    constexpr uint32_t d_share_tiles = get_arg(args::d_share_tiles);
    constexpr uint32_t tensor_pos_tiles = get_arg(args::tensor_pos_tiles);
    constexpr uint32_t clusters_per_user = get_arg(args::clusters_per_user);
    constexpr uint32_t max_workers_per_user = get_arg(args::max_workers_per_user);
    constexpr uint32_t merge_cost = get_arg(args::merge_cost_units);
    constexpr uint32_t pos_page_bytes = get_arg(args::pos_page_bytes);
    constexpr uint32_t hart_scratch = get_arg(args::writer_hart_scratch);
    constexpr uint32_t neos_served = C / W;
    constexpr uint32_t partial_tiles = Dt + 2;
    static_assert(C % W == 0, "a writer hart serves whole NEOs");

    const uint32_t hart = get_my_thread_id();

    DataflowBuffer partial_ring(dfb::partial_out);
    DataflowBuffer output_ring(dfb::output_tiles);
    Noc noc;
    const auto acc_pos = TensorAccessor(tensor::cur_pos);
    const auto acc_partials = TensorAccessor(tensor::partials);
    const auto acc_output = TensorAccessor(tensor::output);
    Scratchpad<uint32_t> scratch(scratch::scratch);
    Semaphore<> partials_ready[] = {
        Semaphore<>(sem::partials_ready_0),
        Semaphore<>(sem::partials_ready_1),
        Semaphore<>(sem::partials_ready_2),
        Semaphore<>(sem::partials_ready_3)};
    constexpr NocOptVals user_txn = {.trid = 0};
    const uint32_t region = hart * hart_scratch;
    const uint32_t partial_bytes = partial_ring.get_tile_size();
    const uint32_t output_bytes = output_ring.get_tile_size();

    for (uint32_t u = user_start; u < user_start + user_count; ++u) {
        noc.async_read<NocOptions::TXN_ID>(
            acc_pos, scratch, pos_page_bytes, {.page_id = 0}, {.offset_bytes = region}, user_txn);
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        uint32_t raw_pos;
        {
            auto lock = scratch.scoped_lock(region / 4, hart_scratch / 4);
            raw_pos = scratch[region / 4 + u];
        }
        // cluster_active and workers_active do not depend on the NEO.
        const auto plan = decode_plan::make_plan<tensor_pos_tiles, kv_heads, C, clusters_per_user, merge_cost>(
            raw_pos, cluster_rank, hart);
        if (!plan.cluster_active) {
            continue;
        }

        // ---- store_partial_block: one served NEO's whole partial per block, one barrier per block. wait/pop act on
        // the tile counter of the NEO being served and pop advances to the next one (dataflow_buffer.inl, DM path),
        // so the rotation is per partial. The ring holds exactly one partial per NEO (capacity partial_tiles), so a
        // block starts at the counter's first entry and never wraps; entries of one counter are stride_size apart.
        const uint32_t partial_stride = partial_ring.get_stride_size();
        for (uint32_t j = 0; j < neos_served; ++j) {
            const uint32_t neo = hart + j * W;
            const uint32_t slot = u * max_workers_per_user + cluster_rank * C + neo;
            partial_ring.wait_front(partial_tiles);
            for (uint32_t i = 0; i < partial_tiles; ++i) {
                noc.async_write<NocOptions::TXN_ID>(
                    partial_ring,
                    acc_partials,
                    partial_bytes,
                    {.offset_bytes = i * partial_stride},
                    {.page_id = slot * partial_tiles + i},
                    user_txn);
            }
            noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
            partial_ring.pop_front(partial_tiles);
        }
        // Every combiner reader hart counts this hart's partials.
        for (uint32_t s = 0; s < C; ++s) {
            partials_ready[s].up(noc, combiner_x, combiner_y, neos_served);
        }
        noc.async_atomic_barrier();

        // ---- store_output_block
        if (cluster_rank == 0) {
            for (uint32_t i = 0; i < d_share_tiles; ++i) {
                for (uint32_t j = 0; j < neos_served; ++j) {
                    const uint32_t neo = hart + j * W;
                    const uint32_t d = i * C + neo < Dt ? i * C + neo : i * C;  // short last round: re-store
                    output_ring.wait_front(1);
                    noc.async_write<NocOptions::TXN_ID>(
                        output_ring, acc_output, output_bytes, {.offset_bytes = 0}, {.page_id = u * Dt + d}, user_txn);
                    noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                    output_ring.pop_front(1);
                }
            }
        }
    }
    noc_async_write_set_trid(0, noc.get_noc_id());
}

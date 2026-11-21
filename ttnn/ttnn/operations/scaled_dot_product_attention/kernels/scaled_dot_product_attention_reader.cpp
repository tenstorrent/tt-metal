// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// scaled_dot_product_attention reader: hart p of R reader harts of a cluster.
//
// query_block (STRIDED, one slice per NEO): hart p feeds NEOs p, p+R, ... (C/R of them), one whole Q block
//   (q_chunk_tiles x Dt tiles of the NEO's unit) per NEO per round, in strict rotation over its NEOs.
// key_block, value_block (ALL: every NEO of the cluster sees every block): the cluster's key chunks are numbered
//   over all its rounds; chunk m is read by hart m mod R into its own producer region, K block then V block.
//   The total is padded to a multiple of R with credit-only chunks (every producer pushes the same count).
//   Causal: only key tile-rows below the round's last query tile + 1 are read (the push stays the nominal
//   block; the compute never reads the rest).
// constants (STRIDED): per NEO the reduce scaler (1.0, row 0 of every face), and when causal the diagonal
//   mask tile (0 where column <= row, -inf above), written once at kernel start.
//
// Explicit sync on every ring: whole blocks assembled from several reads (reserve / read / barrier / push),
// and constant tiles the hart writes itself. Every NoC transfer carries the user transaction id (trid 0), so
// the hart never inherits or leaves a ring's tag (ring protocols).

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/quasar/cache.h"
#endif

#include "sdpa_geometry.hpp"

namespace {

// The causal diagonal mask in tile face order (faces of 16 x 16: f0 rows 0-15 / cols 0-15, f1 rows 0-15 /
// cols 16-31, f2 rows 16-31 / cols 0-15, f3 rows 16-31 / cols 16-31): -inf where column > row, 0 elsewhere,
// in the ring's format (Float32 or bfloat16). The tile is zeroed by the data-movement engine; only the -inf
// lanes are written by the hart.
template <DataFormat format, class Ring>
FORCE_INLINE void push_diag_mask(Ring& ring, Noc& noc) {
    static_assert(format == DataFormat::Float32 || format == DataFormat::Float16_b, "mask format");
    using Elem = std::conditional_t<format == DataFormat::Float32, uint32_t, uint16_t>;
    constexpr Elem NEG_INF = format == DataFormat::Float32 ? Elem(0xFF800000u) : Elem(0xFF80u);
    ring.reserve_back(1);
    const uint32_t addr = ring.get_write_ptr();
    const uint32_t tile_bytes = ring.get_entry_size();
    noc.async_write_zeros(ring, tile_bytes);
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr Elem* tile = reinterpret_cast<volatile tt_l1_ptr Elem*>(dataflow_kernel_lib::addr_to_l1_ptr(addr));
    constexpr uint32_t FACE = 16;
    constexpr uint32_t FACE_ELEMS = FACE * FACE;
    for (uint32_t r = 0; r < FACE; ++r) {
        for (uint32_t c = r + 1; c < FACE; ++c) {
            tile[0 * FACE_ELEMS + r * FACE + c] = NEG_INF;  // f0: above the diagonal
            tile[3 * FACE_ELEMS + r * FACE + c] = NEG_INF;  // f3: above the diagonal
        }
        for (uint32_t c = 0; c < FACE; ++c) {
            tile[1 * FACE_ELEMS + r * FACE + c] = NEG_INF;  // f1: every column is above every row
        }
    }
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // The stores land in this hart's caches; write them through so the NEO's unpacker reads them.
    flush_l2_cache_range(addr, tile_bytes);
#endif
    ring.push_back(1);
}

}  // namespace

void kernel_main() {
    const uint32_t first_round = get_arg(args::first_round);
    const uint32_t num_rounds = get_arg(args::num_rounds);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t Dt = get_arg(args::Dt);
    constexpr uint32_t qc = get_arg(args::q_chunk_tiles);
    constexpr uint32_t kc = get_arg(args::k_chunk_tiles);
    constexpr bool causal = get_arg(args::causal) != 0;
    constexpr uint32_t neos_per_hart = C / R;
    constexpr uint32_t q_block_tiles = qc * Dt;
    constexpr uint32_t kv_block_tiles = kc * Dt;
    using G = SDPA_GEOMETRY_FROM_ARGS;
    static_assert(C % R == 0, "R must divide C");

    const uint32_t p = get_my_thread_id();
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer q_ring(dfb::query_block);
    DataflowBuffer k_ring(dfb::key_block);
    DataflowBuffer v_ring(dfb::value_block);
    DataflowBuffer constants(dfb::constants);
    Noc noc;
    const auto acc_q = TensorAccessor(tensor::q);
    const auto acc_k = TensorAccessor(tensor::k);
    const auto acc_v = TensorAccessor(tensor::v);
    const uint32_t q_tile_bytes = q_ring.get_entry_size();
    const uint32_t kv_tile_bytes = k_ring.get_entry_size();

    // ---- constants: entry k to each of this hart's NEOs before entry k + 1 (strict rotation), in the entry
    // order of sdpa_geometry.hpp.
    static_assert(SCALER_TILE == 0 && DIAG_MASK_TILE == 1, "the pushes below follow the constants' entry order");
    for (uint32_t j = 0; j < neos_per_hart; ++j) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            uint32_t(dfb::constants),
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW>(constants);  // 1.0 on row 0 of every face: also the SUM scaler
    }
    if constexpr (causal) {
        for (uint32_t j = 0; j < neos_per_hart; ++j) {
            push_diag_mask<get_dataformat(dfb::constants)>(constants, noc);
        }
    }

    // One K or V block of key tile-rows [chunk * kc, chunk * kc + rows) into this hart's region.
    auto load_kv_block = [&](auto& ring, const auto& acc, uint32_t b, uint32_t c, uint32_t chunk, uint32_t rows)
        __attribute__((always_inline)) {
            ring.reserve_back(kv_block_tiles);
            const uint32_t stride = ring.get_stride_size();
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc.async_read<NocOptions::TXN_ID>(
                        acc,
                        ring,
                        kv_tile_bytes,
                        {.page_id = G::kv_page(b, c, chunk * kc + r, d)},
                        {.offset_bytes = (r * Dt + d) * stride},
                        user_txn);
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            ring.push_back(kv_block_tiles);
        };
    auto credit_only_block = [&](auto& ring) __attribute__((always_inline)) {
        ring.reserve_back(kv_block_tiles);
        ring.push_back(kv_block_tiles);
    };

    uint32_t chunk_counter = 0;  // the cluster's key chunks over all its rounds
    for (uint32_t r = 0; r < num_rounds; ++r) {
        const auto rc = G::round_coords(first_round + r);

        // ---- Q: one block per NEO this hart feeds.
        for (uint32_t j = 0; j < neos_per_hart; ++j) {
            const uint32_t u = G::unit_of(rc.head_round, p + j * R);
            const uint32_t h = G::q_head(rc.c, u);
            q_ring.reserve_back(q_block_tiles);
            const uint32_t stride = q_ring.get_stride_size();
            for (uint32_t i = 0; i < qc; ++i) {
                const uint32_t row = G::q_row(u, i);
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc.async_read<NocOptions::TXN_ID>(
                        acc_q,
                        q_ring,
                        q_tile_bytes,
                        {.page_id = G::q_page(rc.b, h, row, d)},
                        {.offset_bytes = (i * Dt + d) * stride},
                        user_txn);
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            q_ring.push_back(q_block_tiles);
        }

        // ---- K and V: chunk m of the cluster by hart m mod R.
        const uint32_t key_limit = G::round_key_limit(rc.head_round);
        const uint32_t chunks = G::round_chunks(rc.head_round);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk, ++chunk_counter) {
            if (chunk_counter % R != p) {
                continue;
            }
            const uint32_t rows = G::chunk_extent(chunk, key_limit);
            load_kv_block(k_ring, acc_k, rc.b, rc.c, chunk, rows);
            load_kv_block(v_ring, acc_v, rc.b, rc.c, chunk, rows);
        }
    }
    // ---- credit-only chunks: every producer region of the ALL rings gets the same number of blocks.
    for (; chunk_counter % R != 0; ++chunk_counter) {
        if (chunk_counter % R == p) {
            credit_only_block(k_ring);
            credit_only_block(v_ring);
        }
    }
    // Every transfer above carried the user id; leave it in the tag register for whoever runs next.
    noc_async_read_set_trid(0, noc.get_noc_id());
}

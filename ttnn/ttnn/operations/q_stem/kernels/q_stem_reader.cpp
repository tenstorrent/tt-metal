// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem compute-core reader (RISCV_1 / NoC0). Block ops:
//   qb_slice_load_block  (stage-1 cores) the core's Q_t x cols slice of q_b_proj, once, resident
//   x rounds             (stage-1 cores) receive every x round multicast by the injector (Mcast helper receiver)
//   wkv_load_block       wkv_b1 units into cb_wkv: R1 one head (Dn_t x N_t), R2 one (row, N-block) (Dn_t x s3n);
//                        the first `wkv_depth` units are prefetched before stage 1 (they need no remote data)
//   land_row_block       wait the row's arrival count on the landing scratch, then push credits into cb_landing
//                        (zero-copy: cb_landing IS this core's landing shard); re-arm the counters at the end.
//   rope_tables_block    (RoPE, cores holding heads) the trans tile (rotate_half_meta as a 32x32 0/+-1 matrix,
//                        generated in L1: NoC zero-fill + its 32 non-zeros) and the quarter's cos / sin rows
//                        (qrt x Dr_t tiles each, one DRAM read batch), once, resident for every rope_block.
// Arrival counters arrive[src] count landed tiles. A remote source has exactly one in-order stream into this core
// (one port, link = core index mod L), so a per-row threshold is exact; the own quarter arrives from many stage-1
// writers in any order, so a local head waits for the total of all local tiles before its first row.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/ttnn/operations/q_stem/kernels/q_stem_common.hpp"

using namespace dataflow_kernel_lib;

namespace {
// x rounds of stage 1 (receiver side of `x_round_multicast_block`). Templates so the receiver pipes are only
// instantiated for the stage-1 kernel (the other compute cores carry absent multicast tags).
//
// NI rotating injectors (host X_INJECTORS; one Mcast family each, alternating NoCs): round r comes from injector
// r % NI, as that family's round r / NI. Flow control is ours, not the helper's handshake (which acks a round only
// when the receiver starts waiting for it, so the injectors would take turns): a receiver credits slot r to
// injector r % NI (an atomic inc on its credit semaphore) as soon as the slot is free, up to `ahead` =
// min(NI, x_round_depth) rounds ahead — one outstanding round per injector, so each injector's data-ready flag is
// seen and cleared before that injector is credited again (Flag signal: linked multicast, no per-round barrier).
template <uint32_t NI, typename MC0, typename MC1, typename F>
void receive_x_rounds(
    const MC0& mc0,
    const MC1& mc1,
    uint32_t cb_x_round,
    uint32_t round_tiles,
    uint32_t num_rounds,
    uint32_t ahead,
    const uint64_t* injector_credit,
    F&& after_round) {
    if constexpr (MC0::active) {
        static_assert(NI == 1 || MC1::active, "a second injector needs its multicast family");
        Noc noc;
        auto pipe0 = mc0.receiver(noc);
        uint32_t acked = 0;
        auto credit = [&](uint32_t r, bool block_first) {
            while (acked < num_rounds && acked < r + ahead) {
                const uint32_t pages = (acked - r + 1) * round_tiles;  // slots r .. acked free
                if (block_first && acked == r) {
                    cb_reserve_back(cb_x_round, pages);
                } else if (!cb_pages_reservable_at_back(cb_x_round, pages)) {
                    break;
                }
                noc_semaphore_inc(injector_credit[acked % NI], 1);
                ++acked;
            }
        };
        if constexpr (NI > 1) {
            auto pipe1 = mc1.receiver(noc);
            for (uint32_t r = 0; r < num_rounds; ++r) {
                credit(r, true);
                if (r % NI == 0) {
                    pipe0.receive(r / NI);
                } else {
                    pipe1.receive(r / NI);
                }
                cb_push_back(cb_x_round, round_tiles);
                after_round();
            }
        } else {
            for (uint32_t r = 0; r < num_rounds; ++r) {
                credit(r, true);
                pipe0.receive(r);
                cb_push_back(cb_x_round, round_tiles);
                after_round();
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t cb_qb_slice = get_compile_time_arg_val(1);
    constexpr uint32_t cb_landing = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wkv = get_compile_time_arg_val(3);
    constexpr uint32_t qb_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t wkv_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t Q_t = get_compile_time_arg_val(6);
    constexpr uint32_t Cq_t = get_compile_time_arg_val(7);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t nkb = get_compile_time_arg_val(9);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t Dn_t = get_compile_time_arg_val(11);
    constexpr uint32_t N_t = get_compile_time_arg_val(12);
    constexpr uint32_t s3n = get_compile_time_arg_val(13);
    constexpr uint32_t n_nb = get_compile_time_arg_val(14);
    constexpr bool wkv_resident = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t wkv_depth = get_compile_time_arg_val(16);
    constexpr uint32_t qb_batch = get_compile_time_arg_val(17);  // q_b reads per barrier (host QB_READ_BATCH)
    constexpr uint32_t B = get_compile_time_arg_val(18);
    constexpr uint32_t qrt = get_compile_time_arg_val(19);
    constexpr uint32_t G = get_compile_time_arg_val(21);
    constexpr uint32_t hw = get_compile_time_arg_val(22);
    constexpr uint32_t ts = get_compile_time_arg_val(23);  // stage-3 consumption order (q_stem::RowOrder)
    constexpr auto qb_args = TensorAccessorArgs<24>();
    constexpr auto wkv_args = TensorAccessorArgs<qb_args.next_compile_time_args_offset()>();
    // RoPE tables (cos and sin share one layout; a stand-in accessor when NoPE)
    [[maybe_unused]] constexpr auto cs_args = TensorAccessorArgs<wkv_args.next_compile_time_args_offset()>();
    constexpr bool rope = get_named_compile_time_arg_val("rope") != 0;
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_trans = get_named_compile_time_arg_val("cb_trans");
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast0_ct_offset"),
        get_named_compile_time_arg_val("mcast0_rt_offset")>();
    constexpr auto mc1 = McastArgs<
        get_named_compile_time_arg_val("mcast1_ct_offset"),
        get_named_compile_time_arg_val("mcast1_rt_offset")>();
    constexpr uint32_t x_injectors = get_named_compile_time_arg_val("x_injectors");
    constexpr uint32_t x_round_depth = get_named_compile_time_arg_val("x_round_depth");
    constexpr uint32_t sem_x_credit = get_named_compile_time_arg_val("sem_x_credit");

    constexpr uint32_t rows_per_head = B * qrt;
    constexpr uint32_t unit_tiles = Dn_t * s3n;  // one wkv unit (R1: s3n == N_t, one head)

    size_t a = 0;
    const uint32_t qb_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c0 = get_arg_val<uint32_t>(a++);
    const uint32_t cols = get_arg_val<uint32_t>(a++);
    const uint32_t wkv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t my_pos = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t cos_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t sin_addr = get_arg_val<uint32_t>(a++);
    uint32_t arrive_addr[G];
    for (uint32_t i = 0; i < G; ++i) {
        arrive_addr[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t hpc = get_arg_val<uint32_t>(a++);
    const uint32_t heads_idx = a;
    auto head_of = [&](uint32_t u) { return get_arg_val<uint32_t>(heads_idx + 2 * u); };
    auto src_of = [&](uint32_t u) { return get_arg_val<uint32_t>(heads_idx + 2 * u + 1); };

    uint32_t nh[G];  // my heads per source position
    for (uint32_t i = 0; i < G; ++i) {
        nh[i] = 0;
    }
    for (uint32_t u = 0; u < hpc; ++u) {
        nh[src_of(u)]++;
    }
    // Consumption order (host: consumption_order / landing_slot; device: q_stem::RowOrder): remote rows t < ts
    // row-major across the n_r remote heads (they all land with the same row profile), the local heads head-major,
    // then the late remote rows.
    const uint32_t n_r = hpc - nh[my_pos];
    const q_stem::RowOrder order(n_r, hpc, rows_per_head, ts);

    const auto wkv = TensorAccessor(wkv_args, wkv_addr, wkv_tile_bytes);
    const uint32_t total_units = wkv_resident ? hpc : hpc * rows_per_head * n_nb;
    // Tile w of unit idx -> wkv page. A unit is Dn_t x s3n tiles, row-major (in1 of the stage-3 matmul).
    auto unit_page = [&](uint32_t idx, uint32_t w) {
        uint32_t u, nb;
        if constexpr (wkv_resident) {
            u = idx;  // one unit per head, in consumption order
            nb = 0;
        } else {  // one unit per (row, head, N-block), in consumption order
            uint32_t t;
            order.at(idx / n_nb, u, t);
            nb = idx % n_nb;
        }
        const uint32_t kt = w / s3n;
        return head_of(u) * Dn_t * N_t + nb * s3n + kt * N_t + (w - kt * s3n);
    };
    // Issue the reads of tiles [w0, w1) of the unit sequence starting at `first_unit`, into l1 (no barrier).
    auto issue_tiles = [&](uint32_t first_unit, uint32_t w0, uint32_t w1, uint32_t l1) {
        for (uint32_t w = w0; w < w1; ++w) {
            const uint32_t idx = first_unit + w / unit_tiles;
#ifndef ABLATE_WKV_READ
            noc_async_read(wkv.get_noc_addr(unit_page(idx, w % unit_tiles)), l1 + w * wkv_tile_bytes, wkv_tile_bytes);
#endif
        }
    };

    // ---- stage 1 --------------------------------------------------------------------------------------------------
    uint32_t prefetched = 0;
    if constexpr (mc.active) {
        // qb_slice_load_block: pages k * Cq_t + c0 + c, laid out [k][c] (in1 width = cols). Pushed one K-block
        // (s1_kb rows) at a time: the compute waits per K-block, so the first block's matmul overlaps the rest of
        // the slice read (the slice stays resident; reserved once).
        const auto qb = TensorAccessor(qb_args, qb_addr, qb_tile_bytes);
        constexpr uint32_t s1_kb = Q_t / nkb;
        cb_reserve_back(cb_qb_slice, Q_t * cols);
        uint32_t l1 = get_write_ptr(cb_qb_slice);
        uint32_t n = 0;
        for (uint32_t k = 0; k < Q_t; ++k) {
            for (uint32_t c = 0; c < cols; ++c) {
#ifndef ABLATE_QB_READ
                noc_async_read(qb.get_noc_addr(k * Cq_t + c0 + c), l1, qb_tile_bytes);
#endif
                l1 += qb_tile_bytes;
                if (++n == qb_batch) {
                    noc_async_read_barrier();
                    n = 0;
                }
            }
            if ((k + 1) % s1_kb == 0) {
                noc_async_read_barrier();
                n = 0;
                cb_push_back(cb_qb_slice, s1_kb * cols);
            }
        }
    }

    // wkv prefetch (`wkv_load_block` for the first units): they need no remote data. On a stage-1 core the reads are
    // spread over the x rounds (stage 1 leaves the DRAM idle, and a burst here would slow every core's q_b read,
    // which the lockstep x multicast waits for); the barrier + push come after the last round.
    // Cores outside the stage-1 rect prefetch nothing: at t=0 the DRAM is busy with the q_b slices every stage-1
    // core waits for; they load each head's weights once its first row has landed (below).
    prefetched = mc.active ? (total_units < wkv_depth ? total_units : wkv_depth) : 0;
    const uint32_t prefetch_tiles = prefetched * unit_tiles;
    uint32_t prefetch_l1 = 0, issued = 0;
    if (prefetched > 0) {
        cb_reserve_back(cb_wkv, prefetch_tiles);
        prefetch_l1 = get_write_ptr(cb_wkv);
    }
    constexpr uint32_t num_rounds = num_blocks * nkb;
    const uint32_t per_round = mc.active ? (prefetch_tiles + num_rounds - 1) / num_rounds : prefetch_tiles;
    uint64_t injector_credit[x_injectors];  // credit semaphore of every injector (packed virtual x/y in named args)
    for (uint32_t i = 0; i < x_injectors; ++i) {
        const uint32_t xy =
            i == 0 ? get_named_compile_time_arg_val("x_inj0_xy") : get_named_compile_time_arg_val("x_inj1_xy");
        injector_credit[i] = get_noc_addr(q_stem::unpack_x(xy), q_stem::unpack_y(xy), get_semaphore(sem_x_credit));
    }
    receive_x_rounds<x_injectors>(
        mc,
        mc1,
        cb_x_round,
        round_tiles,
        num_rounds,
        x_injectors < x_round_depth ? x_injectors : x_round_depth,
        injector_credit,
        [&]() {
            const uint32_t to = issued + per_round < prefetch_tiles ? issued + per_round : prefetch_tiles;
            issue_tiles(0, issued, to, prefetch_l1);
            issued = to;
        });
    if (prefetched > 0) {
        issue_tiles(0, issued, prefetch_tiles, prefetch_l1);
        noc_async_read_barrier();
        cb_push_back(cb_wkv, prefetch_tiles);
    }

    // ---- stage 3: land rows, stream the remaining wkv units ---------------------------------------------------------
    // rope_tables_block: trans tile now (local L1 only); cos / sin with the first landed row (below), so a core
    // outside the stage-1 rect does not read them while the DRAM serves the q_b prologue.
    constexpr uint32_t Dr_t = hw - Dn_t;
    constexpr uint32_t cs_tiles = qrt * Dr_t;
    if constexpr (rope) {
        if (hpc > 0) {
            ::DataflowBuffer trans(cb_trans);
            trans.reserve_back(1);
            dataflow_kernel_lib::zero_tile(trans);
            // rot = x @ T, rot[2i] = -x[2i+1], rot[2i+1] = x[2i]: T[2i][2i+1] = +1, T[2i+1][2i] = -1 (bf16). Block
            // diagonal within 16x16 faces: rows/cols 0..15 in face 0, 16..31 in face 3. Element (r, c) of a face at
            // r * 16 + c.
            volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_trans));
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint16_t* face = t + f * 3 * 256;
                for (uint32_t i = 0; i < 8; ++i) {
                    face[(2 * i) * 16 + 2 * i + 1] = 0x3F80;  // +1.0
                    face[(2 * i + 1) * 16 + 2 * i] = 0xBF80;  // -1.0
                }
            }
            trans.push_back(1);
        }
    }
    auto load_rope_tables = [&]() {
        if constexpr (rope) {
            constexpr uint32_t cs_bytes = get_tile_size(cb_cos);
            const auto cos_t = TensorAccessor(cs_args, cos_addr, cs_bytes);
            const auto sin_t = TensorAccessor(cs_args, sin_addr, cs_bytes);
            cb_reserve_back(cb_cos, cs_tiles);
            cb_reserve_back(cb_sin, cs_tiles);
            uint32_t lc = get_write_ptr(cb_cos), ls = get_write_ptr(cb_sin);
            for (uint32_t i = 0; i < cs_tiles; ++i) {  // pages r * Dr_t + c of (1, 1, Sq, Dr), row tiles r < qrt
                noc_async_read(cos_t.get_noc_addr(i), lc, cs_bytes);
                noc_async_read(sin_t.get_noc_addr(i), ls, cs_bytes);
                lc += cs_bytes;
                ls += cs_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_cos, cs_tiles);
            cb_push_back(cb_sin, cs_tiles);
        }
    };
    uint32_t next_unit = prefetched;
    auto load_next_unit = [&]() {
        if (next_unit < total_units) {
            cb_reserve_back(cb_wkv, unit_tiles);
            issue_tiles(next_unit, 0, unit_tiles, get_write_ptr(cb_wkv));
            noc_async_read_barrier();
            cb_push_back(cb_wkv, unit_tiles);
            ++next_unit;
        }
    };
    auto ensure_units = [&](uint32_t upto) {
        while (next_unit < upto) {
            load_next_unit();
        }
    };
    // Rows in consumption order. Remote row (u, t): wait the row's arrival threshold (one in-order stream per source).
    // R1: every remote head's weights resident before the first remote row (a core outside the stage-1 rect loads
    // them once that row has landed), a local head's weights before its first row; R2: the row's N-block units right
    // after its credit. The own quarter comes from many stage-1 writers in any order, so wait for all of it once,
    // before the first local row.
    for (uint32_t c = 0; c < order.total(); ++c) {
        uint32_t u, t;
        order.at(c, u, t);
        if (u < n_r) {
            const uint32_t src = src_of(u);
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr[src]), (t + 1) * nh[src] * hw);
            if constexpr (wkv_resident) {
                if (c == 0) {
                    ensure_units(n_r);
                }
            }
        } else {
            if (c == order.first_local()) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr[my_pos]),
                    nh[my_pos] * rows_per_head * hw);
            }
            if constexpr (wkv_resident) {
                if (t == 0) {
                    ensure_units(u + 1);
                }
            }
        }
        if (c == 0) {
            load_rope_tables();
        }
        cb_reserve_back(cb_landing, hw);
        cb_push_back(cb_landing, hw);  // credits only: the row is already in my landing shard
        if constexpr (!wkv_resident) {
            ensure_units((c + 1) * n_nb);
        }
    }

    // Re-arm: consume exactly what arrived (every wait above was cumulative).
    for (uint32_t i = 0; i < G; ++i) {
        if (nh[i] > 0) {
            noc_semaphore_inc(get_noc_addr(arrive_addr[i]), 0u - nh[i] * rows_per_head * hw);
        }
    }
    noc_async_atomic_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem compute-core reader (RISCV_1 / NoC0). Block ops:
//   qb_slice_load_block  (stage-1 cores) the core's Q_t x cols slice of q_b_proj, once, resident
//   x rounds             (stage-1 cores) receive every x round multicast by the injector (Mcast helper receiver)
//   wkv_load_block       wkv_b1 units into cb_wkv: R1 one head (Dn_t x N_t), R2 one (row, N-block) (Dn_t x s3n);
//                        the first `wkv_depth` units are prefetched before stage 1 (they need no remote data)
//   land_row_block       wait the row's arrival count on the landing scratch, then push credits into cb_landing
//                        (zero-copy: cb_landing IS this core's landing shard); re-arm the counters at the end.
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

using namespace dataflow_kernel_lib;

namespace {
// x rounds of stage 1 (receiver side of `x_round_multicast_block`). A template so the receiver pipe is only
// instantiated for the stage-1 kernel (the other compute cores carry an absent multicast tag).
template <typename MC, typename F>
void receive_x_rounds(const MC& mc, uint32_t cb_x_round, uint32_t round_tiles, uint32_t num_rounds, F&& after_round) {
    if constexpr (MC::active) {
        Noc noc;
        auto pipe = mc.receiver(noc);
        for (uint32_t r = 0; r < num_rounds; ++r) {
            cb_reserve_back(cb_x_round, round_tiles);  // slot free -> the write ptr is the multicast destination
            pipe.receive();                            // ack (slot free), wait for the round
            cb_push_back(cb_x_round, round_tiles);
            after_round();
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
    constexpr auto qb_args = TensorAccessorArgs<23>();
    constexpr auto wkv_args = TensorAccessorArgs<qb_args.next_compile_time_args_offset()>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

    constexpr uint32_t rows_per_head = B * qrt;
    constexpr uint32_t unit_tiles = Dn_t * s3n;  // one wkv unit (R1: s3n == N_t, one head)

    size_t a = 0;
    const uint32_t qb_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c0 = get_arg_val<uint32_t>(a++);
    const uint32_t cols = get_arg_val<uint32_t>(a++);
    const uint32_t wkv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t my_pos = get_arg_val<uint32_t>(a++);
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
    // Consumption order (host: consumption_order / landing_slot): the n_r remote heads first, row-major across them
    // (they all land with the same row profile), then the local heads head-major.
    const uint32_t n_r = hpc - nh[my_pos];

    const auto wkv = TensorAccessor(wkv_args, wkv_addr, wkv_tile_bytes);
    const uint32_t total_units = wkv_resident ? hpc : hpc * rows_per_head * n_nb;
    // Tile w of unit idx -> wkv page. A unit is Dn_t x s3n tiles, row-major (in1 of the stage-3 matmul).
    auto unit_page = [&](uint32_t idx, uint32_t w) {
        uint32_t u, nb;
        if constexpr (wkv_resident) {
            u = idx;  // one unit per head, in consumption order
            nb = 0;
        } else {  // one unit per (row, head, N-block), in consumption order
            const uint32_t a_units = n_r * rows_per_head * n_nb;
            if (idx < a_units) {
                u = (idx % (n_r * n_nb)) / n_nb;
            } else {
                u = n_r + (idx - a_units) / (rows_per_head * n_nb);
            }
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
        // qb_slice_load_block: pages k * Cq_t + c0 + c, laid out [k][c] (in1 width = cols)
        const auto qb = TensorAccessor(qb_args, qb_addr, qb_tile_bytes);
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
        }
        noc_async_read_barrier();
        cb_push_back(cb_qb_slice, Q_t * cols);
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
    receive_x_rounds(mc, cb_x_round, round_tiles, num_rounds, [&]() {
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
    // Phase A: remote heads, row-major. R1: every remote head's weights resident before the first row (a core outside
    // the stage-1 rect loads them once that row has landed); R2: the row's N-block units right after its credit.
    for (uint32_t t = 0; t < rows_per_head; ++t) {
        for (uint32_t u = 0; u < n_r; ++u) {
            const uint32_t src = src_of(u);
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr[src]), (t + 1) * nh[src] * hw);
            if constexpr (wkv_resident) {
                if (t == 0 && u == 0) {
                    ensure_units(n_r);
                }
            }
            cb_reserve_back(cb_landing, hw);
            cb_push_back(cb_landing, hw);  // credits only: the row is already in my landing shard
            if constexpr (!wkv_resident) {
                ensure_units(((t * n_r) + u + 1) * n_nb);
            }
        }
    }
    // Phase B: local heads, head-major; the own quarter comes from many stage-1 writers in any order, so wait for
    // all of it once.
    if (n_r < hpc) {
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr[my_pos]), nh[my_pos] * rows_per_head * hw);
    }
    for (uint32_t u = n_r; u < hpc; ++u) {
        if constexpr (wkv_resident) {
            ensure_units(u + 1);
        }
        for (uint32_t t = 0; t < rows_per_head; ++t) {
            cb_reserve_back(cb_landing, hw);
            cb_push_back(cb_landing, hw);
            if constexpr (!wkv_resident) {
                ensure_units(n_r * rows_per_head * n_nb + ((u - n_r) * rows_per_head + t + 1) * n_nb);
            }
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

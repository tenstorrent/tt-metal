// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-reader (sender core NCRISC) — perf experiment x_read_fast.
// Reads this sender's x rounds (block_m_tiles x block_k_tiles tiles, (m, k) row-major) into their slots of
// cb_x_resident (round r at base + r * round_bytes) and pushes them in order.
//
// Variants (defines, all default off = the op's current schedule):
//   X_SPLIT_NOC   tile t of a round goes out on NoC (t's k index & 1 ? other NoC : own NoC). The sender BRISC
//                 only WRITES on the other NoC (write cmd buf 0); this RISC uses the read cmd buf (1) there,
//                 which BRISC never touches, and both NoCs' read cmd bufs are programmed by noc_init.
//                 Tagged with the same trid on both NoCs; a round is done when its trid drains on both.
//   X_FAST_ISSUE  raw one-packet stateful reads: per-bank base NoC addresses precomputed once from the
//                 TensorAccessor (page p = bank p % NB, slot p / NB), coordinate/length programmed only when
//                 the bank changes (noc_async_read_one_packet_set_state), then only src/dst + go per tile.
//                 Bypasses noc_async_read_tile / TensorAccessor::get_noc_addr per tile (ergonomics helper, not a
//                 kernel_lib helper): its per-call bank math + full command programming is the issue cost.
//   X_EAGER       retire (push) rounds as soon as their trid drains, polled between issues, instead of only when
//                 the trid is reused — lets a wider trid window keep round 0 early.
// Zones (permanent): x_read_issue, x_read_drain. ABLATE_X_READ skips the DRAM reads (trid set/barrier + pushes kept).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

#ifndef X_NUM_BANKS
#define X_NUM_BANKS 8
#endif

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_read_trids = get_compile_time_arg_val(5);  // rounds in flight
    constexpr uint32_t first_round = get_compile_time_arg_val(6);
    constexpr uint32_t round_stride = get_compile_time_arg_val(7);  // number of x senders
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr uint32_t my_rounds = (num_k_blocks - first_round + round_stride - 1) / round_stride;

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    static_assert(x_read_trids >= 1 && x_read_trids <= NOC_MAX_TRANSACTION_ID, "x_read_trids out of trid range");
    static_assert(round_tiles <= NOC_MAX_TRANSACTION_ID_COUNT, "round exceeds the per-trid pending-read counter");

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const uint32_t round_bytes = round_tiles * x_tile_bytes;
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);

    constexpr uint8_t noc_a = noc_index;
#ifdef X_SPLIT_NOC
    constexpr uint8_t noc_b = 1 - noc_index;
    // This RISC never ran noc_local_state_init on noc_b: resync its issued-read counter (no reads in flight there).
    noc_reads_num_issued[noc_b] = NOC_STATUS_READ_REG(noc_b, NIU_MST_RD_RESP_RECEIVED);
    auto noc_of = [](uint32_t kk) -> uint8_t { return (kk & 1) ? noc_b : noc_a; };
#else
    constexpr uint8_t noc_b = noc_a;
    auto noc_of = [](uint32_t) -> uint8_t { return noc_a; };
#endif

#ifdef X_FAST_ISSUE
    constexpr uint32_t NB = X_NUM_BANKS;
    uint32_t bank_lo[2][NB];
    uint64_t bank_full[2][NB];
    for (uint32_t b = 0; b < NB; ++b) {
        bank_full[0][b] = x_acc.get_noc_addr(b, 0, noc_a);
        bank_lo[0][b] = (uint32_t)bank_full[0][b];
        bank_full[1][b] = x_acc.get_noc_addr(b, 0, noc_b);
        bank_lo[1][b] = (uint32_t)bank_full[1][b];
    }
    uint32_t cur_bank[2] = {0xFFFFFFFF, 0xFFFFFFFF};
#endif

    cb_reserve_back(cb_x_resident, round_tiles * my_rounds);
    const uint32_t base = get_write_ptr(cb_x_resident);

    auto issue_round = [&](uint32_t r) {
#ifdef ABLATE_X_READ
        return;
#endif
        const uint32_t rbase = base + r * round_bytes;
        for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
            const uint8_t noc = noc_of(kk);
            uint32_t dst = rbase + kk * x_tile_bytes;
            uint32_t page = r * block_k_tiles + kk;
            for (uint32_t m = 0; m < block_m_tiles; ++m) {
#ifdef X_FAST_ISSUE
                const uint32_t ni = (noc == noc_a) ? 0 : 1;
                const uint32_t b = page % NB;
                if (b != cur_bank[ni]) {
                    noc_async_read_one_packet_set_state(bank_full[ni][b], x_tile_bytes, 0, noc);
                    cur_bank[ni] = b;
                }
                noc_async_read_one_packet_with_state(bank_lo[ni][b] + (page / NB) * x_tile_bytes, dst, 0, noc);
#else
                noc_async_read(x_acc.get_noc_addr(page, 0, noc), dst, x_tile_bytes, noc);
#endif
                dst += block_k_tiles * x_tile_bytes;
                page += tensor_k_tiles;
            }
        }
    };
    auto set_trid = [&](uint32_t trid) {
        noc_async_read_set_trid(trid, noc_a);
        if constexpr (noc_b != noc_a) {
            noc_async_read_set_trid(trid, noc_b);
        }
    };
    auto trid_of = [](uint32_t i) { return (i % x_read_trids) + 1; };
    auto wait_round = [&](uint32_t i) {
        noc_async_read_barrier_with_trid(trid_of(i), noc_a);
        if constexpr (noc_b != noc_a) {
            noc_async_read_barrier_with_trid(trid_of(i), noc_b);
        }
    };
    auto round_landed = [&](uint32_t i) {
        bool done = ncrisc_noc_read_with_transaction_id_flushed(noc_a, trid_of(i));
        if constexpr (noc_b != noc_a) {
            done = done && ncrisc_noc_read_with_transaction_id_flushed(noc_b, trid_of(i));
        }
        return done;
    };

#ifdef X_PROBE_CHUNK
    // PERF PROBE ONLY (wrong layout): read every bank's slots as contiguous X_PROBE_CHUNK-tile chunks, untagged,
    // one barrier, then push everything. Measures the single-core DRAM read rate vs request size.
    {
        MaybeDeviceZoneScope("x_read_issue");
        constexpr uint32_t total = num_k_blocks * round_tiles;
        constexpr uint32_t per_bank = total / X_NUM_BANKS;
        for (uint32_t s = 0; s < per_bank; s += X_PROBE_CHUNK) {
            for (uint32_t b = 0; b < X_NUM_BANKS; ++b) {
                const uint32_t n = (per_bank - s) < X_PROBE_CHUNK ? (per_bank - s) : X_PROBE_CHUNK;
                const uint8_t noc = ((b & 1) && (noc_b != noc_a)) ? noc_b : noc_a;
                noc_async_read(
                    x_acc.get_noc_addr(b + s * X_NUM_BANKS, 0, noc),
                    base + (b * per_bank + s) * x_tile_bytes,
                    n * x_tile_bytes,
                    noc);
            }
        }
    }
    {
        MaybeDeviceZoneScope("x_read_drain");
        noc_async_read_barrier(noc_a);
        if constexpr (noc_b != noc_a) {
            noc_async_read_barrier(noc_b);
        }
    }
    cb_push_back(cb_x_resident, round_tiles * my_rounds);
    return;
#endif
#ifdef X_PROBE_UNTAGGED
    // PERF PROBE ONLY: real layout, every round untagged, one barrier, then push everything.
    {
        MaybeDeviceZoneScope("x_read_issue");
        for (uint32_t i = 0; i < my_rounds; ++i) {
#ifdef X_PROBE_TAGGED
            set_trid((i % 15) + 1);
#endif
            issue_round(first_round + i * round_stride);
        }
    }
    {
        MaybeDeviceZoneScope("x_read_drain");
        noc_async_read_barrier(noc_a);
        if constexpr (noc_b != noc_a) {
            noc_async_read_barrier(noc_b);
        }
    }
    set_trid(0);
    cb_push_back(cb_x_resident, round_tiles * my_rounds);
    return;
#endif
#ifdef X_HELPER
    // A helper core reads the odd rounds and writes them here, publishing a count in this core's cell
    // (semaphore X_HELPER_SEM). This RISC reads the even rounds (x_read_trids of them in flight) and pushes all
    // rounds in order. Single-sender schedule only (first_round 0, round_stride 1).
    static_assert(first_round == 0 && round_stride == 1, "X_HELPER needs the one-sender schedule");
    {
        constexpr uint32_t n_own = (my_rounds + 1) / 2;
        volatile tt_l1_ptr uint32_t* hcell =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(X_HELPER_SEM));
        uint32_t issued = 0, own_retired = 0;
        auto top_up = [&]() {
            while (issued < n_own && issued < own_retired + x_read_trids) {
                set_trid(trid_of(issued));
                issue_round(2 * issued);
                ++issued;
            }
        };
        MaybeDeviceZoneScope("x_read_issue");
        for (uint32_t g = 0; g < my_rounds; ++g) {
            top_up();
            if ((g & 1) == 0) {
                wait_round(own_retired);
                ++own_retired;
                top_up();
            } else {
                while (*hcell < (g + 1) / 2) {
                }
            }
            cb_push_back(cb_x_resident, round_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("x_read_drain");
        noc_async_read_barrier(noc_a);
        if constexpr (noc_b != noc_a) {
            noc_async_read_barrier(noc_b);
        }
    }
    set_trid(0);
    return;
#endif
    uint32_t retired = 0;
    {
        MaybeDeviceZoneScope("x_read_issue");
        for (uint32_t i = 0; i < my_rounds; ++i) {
            if (i - retired == x_read_trids) {
                wait_round(retired);  // its trid is about to be reused
                cb_push_back(cb_x_resident, round_tiles);
                ++retired;
            }
#ifdef X_EAGER
            while (retired < i && round_landed(retired)) {
                cb_push_back(cb_x_resident, round_tiles);
                ++retired;
            }
#endif
            set_trid(trid_of(i));
            issue_round(first_round + i * round_stride);
        }
    }
    MaybeDeviceZoneScope("x_read_drain");
    for (; retired < my_rounds; ++retired) {
        wait_round(retired);
        cb_push_back(cb_x_resident, round_tiles);
    }
    noc_async_read_barrier(noc_a);
    if constexpr (noc_b != noc_a) {
        noc_async_read_barrier(noc_b);
    }
    set_trid(0);
}

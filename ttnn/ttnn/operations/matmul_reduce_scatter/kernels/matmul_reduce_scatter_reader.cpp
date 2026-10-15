// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — matmul-core reader (NCRISC, NoC0).
//
// Per entry (transport order): load_operand_block — Kt/kb K-block windows of A (Mcast1D family "a", one sender per
// n-line) and W (family "b", one sender per m-line). The cross-block operand is resident after entry 0: its windows are
// re-credited (reserve + push, no NoC) because the CB holds the whole Kt slice and the ring wraps exactly.
// Then land_arrivals_block for the lagged (pending) entry: wait the global arrival counter of each arriving direction
// until it covers this entry (host threshold), credit the zone slot (the upstream port's fabric writes already landed
// the bytes). At the end: re-arm both counters by decrementing what arrived.
//
// Raw-API note (design "Helpers considered and rejected"): land_arrivals_block is a semaphore wait + CB credit of
// bytes a remote chip wrote; no helper covers a fabric fused-inc counter with host thresholds.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(3);
    constexpr uint32_t core_m = get_compile_time_arg_val(4);
    constexpr uint32_t core_n = get_compile_time_arg_val(5);
    constexpr uint32_t kb = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t Kt = get_compile_time_arg_val(8);
    constexpr uint32_t Nt = get_compile_time_arg_val(9);
    constexpr uint32_t a_row_per_block = get_compile_time_arg_val(10);  // scatter -2: block_m, else 0
    constexpr uint32_t w_col_per_block = get_compile_time_arg_val(11);  // scatter -1: block_n, else 0
    constexpr bool resident_a = get_compile_time_arg_val(12) != 0;
    constexpr bool resident_w = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t P = get_compile_time_arg_val(14);
    constexpr uint32_t num_entries = get_compile_time_arg_val(15);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(17);
    constexpr auto a_args = TensorAccessorArgs<18>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto mc_a =
        McastArgs<get_named_compile_time_arg_val("a_ct_offset"), get_named_compile_time_arg_val("a_rt_offset")>();
    constexpr auto mc_b =
        McastArgs<get_named_compile_time_arg_val("b_ct_offset"), get_named_compile_time_arg_val("b_rt_offset")>();

    constexpr uint32_t a_window_tiles = core_m * kb;
    constexpr uint32_t w_window_tiles = kb * core_n;

    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);  // gm * core_m (block-local)
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);  // gn * core_n (block-local)
    const uint32_t m_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t n_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t arr_fwd_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t arr_bwd_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t total_fwd = get_arg_val<uint32_t>(arg++);
    const uint32_t total_bwd = get_arg_val<uint32_t>(arg++);
    const uint32_t entries_idx = arg;  // per entry: [block | arrival_mask << 16, thr_fwd | thr_bwd << 16]

    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    volatile tt_l1_ptr uint32_t* arr_fwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_fwd_addr);
    volatile tt_l1_ptr uint32_t* arr_bwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_bwd_addr);

    Noc noc;
    auto a_tx = mc_a.optional_sender(noc);
    auto a_rx = mc_a.optional_receiver(noc);
    auto w_tx = mc_b.optional_sender(noc);
    auto w_rx = mc_b.optional_receiver(noc);
    const bool a_sender = a_tx.has_value();
    const bool w_sender = w_tx.has_value();

    // load_operand_block: Kt/kb windows of both operands for one entry (block b).
    auto load_operand_block = [&](uint32_t entry_idx, uint32_t block) {
        const bool recredit_a = resident_a && entry_idx > 0;
        const bool recredit_w = resident_w && entry_idx > 0;
        const uint32_t a_row_base = block * a_row_per_block + row0;
        const uint32_t w_col_base = block * w_col_per_block + col0;
        for (uint32_t kblk = 0; kblk < num_k_blocks; ++kblk) {
            const uint32_t k0 = kblk * kb;
            cb_reserve_back(cb_act_k_window, a_window_tiles);
            cb_reserve_back(cb_weight_k_window, w_window_tiles);
            const uint32_t a_dst = get_write_ptr(cb_act_k_window);
            const uint32_t w_dst = get_write_ptr(cb_weight_k_window);
            bool issued = false;
            if (!recredit_a && a_sender) {
                for (uint32_t i = 0; i < m_valid; ++i) {
                    const uint32_t page = (a_row_base + i) * Kt + k0;
                    for (uint32_t kk = 0; kk < kb; ++kk) {
                        noc_async_read(
                            a_acc.get_noc_addr(page + kk), a_dst + (i * kb + kk) * a_tile_bytes, a_tile_bytes);
                    }
                }
                issued = true;
            }
            if (!recredit_w && w_sender) {
                for (uint32_t kk = 0; kk < kb; ++kk) {
                    const uint32_t page = (k0 + kk) * Nt + w_col_base;
                    for (uint32_t j = 0; j < n_valid; ++j) {
                        noc_async_read(
                            w_acc.get_noc_addr(page + j), w_dst + (kk * core_n + j) * w_tile_bytes, w_tile_bytes);
                    }
                }
                issued = true;
            }
            if (issued) {
                noc_async_read_barrier();  // one barrier per K-block for both operands
            }
            if (!recredit_a) {
                if (a_sender) {
                    a_tx->send(a_dst, a_dst, a_window_tiles * a_tile_bytes);
                } else {
                    a_rx->receive();
                }
            }
            if (!recredit_w) {
                if (w_sender) {
                    w_tx->send(w_dst, w_dst, w_window_tiles * w_tile_bytes);
                } else {
                    w_rx->receive();
                }
            }
            cb_push_back(cb_act_k_window, a_window_tiles);
            cb_push_back(cb_weight_k_window, w_window_tiles);
        }
    };

    // land_arrivals_block: the zone slot(s) of this entry are credited once the counter covers them.
    auto land_arrivals_block = [&](uint32_t entry_idx) {
        const uint32_t word0 = get_arg_val<uint32_t>(entries_idx + 2 * entry_idx);
        const uint32_t word1 = get_arg_val<uint32_t>(entries_idx + 2 * entry_idx + 1);
        const uint32_t mask = word0 >> 16;
        if (mask & 1) {
            noc_semaphore_wait_min(arr_fwd, word1 & 0xFFFF);
            cb_reserve_back(cb_arrivals_fwd, P);
            cb_push_back(cb_arrivals_fwd, P);
        }
        if (mask & 2) {
            noc_semaphore_wait_min(arr_bwd, word1 >> 16);
            cb_reserve_back(cb_arrivals_bwd, P);
            cb_push_back(cb_arrivals_bwd, P);
        }
    };
    auto needs_arrival = [&](uint32_t entry_idx) {
        return (get_arg_val<uint32_t>(entries_idx + 2 * entry_idx) >> 16) != 0;
    };

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    for (uint32_t e = 0; e < num_entries; ++e) {
        const uint32_t block = get_arg_val<uint32_t>(entries_idx + 2 * e) & 0xFFFF;
        load_operand_block(e, block);
        if (pending != NONE) {
            land_arrivals_block(pending);
            pending = NONE;
        }
        if (needs_arrival(e) && e + 1 < num_entries) {
            pending = e;  // stall shadow: the add waits behind the next entry's matmul
        } else {
            land_arrivals_block(e);
        }
    }

    // Re-arm the global arrival counters for the next call (decrement what arrived, never store 0).
    if (total_fwd > 0) {
        noc_semaphore_wait_min(arr_fwd, total_fwd);
        noc_semaphore_inc(get_noc_addr(arr_fwd_addr), 0u - total_fwd);
    }
    if (total_bwd > 0) {
        noc_semaphore_wait_min(arr_bwd, total_bwd);
        noc_semaphore_inc(get_noc_addr(arr_bwd_addr), 0u - total_bwd);
    }
    noc_async_atomic_barrier();
}

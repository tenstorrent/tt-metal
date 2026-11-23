// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-reader (sender core, the RISC/NoC opposite the multicaster): read this sender's x
// rounds r = first_round, first_round + round_stride, ... (block_m_tiles x block_k_tiles tiles each,
// (m, k) row-major) into their slots of cb_x_resident and push each round as soon as it has landed.
// The multicast kernel on the same core consumes the rounds in the same order. With several x
// senders each owns every round_stride-th round; round r always lands at base + r * round_bytes,
// the address it has on every worker.
//
// cb_x_resident holds the whole x, so every round's slot is known up front (base + r * round_bytes)
// and the reads of up to `x_read_trids` rounds stay in flight at once: round r is tagged with
// transaction id (r % x_read_trids) + 1 and retired with a barrier on that id only, so the NoC never
// drains between rounds (master.md `trid_double_issue`). x_read_trids == 1 is the old
// one-round-then-full-barrier schedule.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

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
    const Noc noc;

    // Whole x is resident: reserve this sender's rounds once, push round by round in order.
    cb_reserve_back(cb_x_resident, round_tiles * my_rounds);
    const uint32_t base = get_write_ptr(cb_x_resident);

    auto issue_round = [&](uint32_t r) {
        uint32_t dst = base + r * round_bytes;
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            const uint32_t row_page = m * tensor_k_tiles + r * block_k_tiles;
            for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
                noc_async_read_tile(row_page + kk, x_acc, dst);
                dst += x_tile_bytes;
            }
        }
    };

    for (uint32_t i = 0; i < my_rounds; ++i) {
        const uint32_t trid = (i % x_read_trids) + 1;
        if (i >= x_read_trids) {
            // Local round i - x_read_trids carries this id: retire it (in order) before reusing the id.
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
            cb_push_back(cb_x_resident, round_tiles);
        }
        dataflow_kernel_lib::set_read_trid(noc, trid);
        issue_round(first_round + i * round_stride);
    }
    constexpr uint32_t to_drain = my_rounds < x_read_trids ? my_rounds : x_read_trids;
    for (uint32_t d = 0; d < to_drain; ++d) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, ((my_rounds - to_drain + d) % x_read_trids) + 1);
        cb_push_back(cb_x_resident, round_tiles);
    }
    noc_async_read_barrier();
    dataflow_kernel_lib::set_read_trid(noc, 0);
}

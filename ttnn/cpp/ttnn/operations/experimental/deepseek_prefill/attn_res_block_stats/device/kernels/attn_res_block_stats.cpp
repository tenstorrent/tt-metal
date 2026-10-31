// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Both statistics of a candidate's token row-tile, into one accumulator:
//
//   dst  = sum_k v_k @ Q_k           the dots, in Q's columns 1..R; column 0 of Q is zero
//   dst += sum_k (v_k * v_k) @ E     the sum of squares, E being ones in column 0
//
// The squares have to reach a circular buffer before a matmul can read them, so they are
// produced first, a run at a time, and the two matmul runs then share one dest session.
//
// A core takes a run of a row (see attn_res_block_stats_items.hpp), so where a row is cut,
// these are partial sums over the run: a run's core hands them to the row's leader, whose
// own go to `cb_local`, and the leader sums the row's in fp32 dest.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/attn_res_block_stats_items.hpp"

using namespace ckernel;

namespace {

constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
constexpr uint32_t kSplitArgs = 1;
using Split = AttnResBlockStatsSplit<kSplitArgs>;
constexpr uint32_t max_run = Split::max_run;
constexpr uint32_t chunks_per_row = Split::chunks_per_row;

constexpr auto cb_v = tt::CBIndex::c_0;
constexpr auto cb_queries = tt::CBIndex::c_1;
constexpr auto cb_ones = tt::CBIndex::c_2;
constexpr auto cb_squares = tt::CBIndex::c_3;
constexpr auto cb_local = tt::CBIndex::c_4;     // a row leader's own partials
constexpr auto cb_received = tt::CBIndex::c_5;  // a row leader's: its other runs' partials
constexpr auto cb_out = tt::CBIndex::c_16;

// One candidate's statistics over a run of `num_tiles`, packed to `out`.
void run_statistics(uint32_t num_tiles, uint32_t out) {
    CircularBuffer cb_v_obj(cb_v);
    CircularBuffer cb_squares_obj(cb_squares);
    CircularBuffer cb_out_obj(out);

    cb_v_obj.wait_front(max_run);

    cb_squares_obj.reserve_back(max_run);
    mul_init(cb_v, cb_v);
    pack_reconfig_data_format(cb_squares);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        tile_regs_acquire();
        mul_tiles(cb_v, cb_v, k, k, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_squares, k);
        tile_regs_release();
    }
    cb_squares_obj.push_back(max_run);
    cb_squares_obj.wait_front(max_run);

    cb_out_obj.reserve_back(1);
    tile_regs_acquire();
    matmul_init(cb_v, cb_queries);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        matmul_tiles(cb_v, cb_queries, k, k, 0);
    }
    matmul_init(cb_squares, cb_ones);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        matmul_tiles(cb_squares, cb_ones, k, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(out);
    pack_tile(0, out);
    tile_regs_release();
    cb_out_obj.push_back(1);

    cb_squares_obj.pop_front(max_run);
    cb_v_obj.pop_front(max_run);
}

// A row leader's: each candidate's own partial plus the row's other runs', which arrive
// run-major.
void sum_row() {
    CircularBuffer cb_local_obj(cb_local);
    CircularBuffer cb_received_obj(cb_received);
    CircularBuffer cb_out_obj(cb_out);

    cb_local_obj.wait_front(num_candidates);
    cb_received_obj.wait_front((chunks_per_row - 1) * num_candidates);
    // The partials are fp32 where the runs were bf16: the copy's init does not move the
    // unpacker off the format the matmuls left it in.
    reconfig_data_format_srca(cb_v, cb_local);
    copy_init(cb_local);
    pack_reconfig_data_format(cb_out);
    for (uint32_t c = 0; c < num_candidates; ++c) {
        cb_out_obj.reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_local, c, 0);
        add_binary_tile_init();
        for (uint32_t r = 1; r < chunks_per_row; ++r) {
            copy_tile(cb_received, (r - 1) * num_candidates + c, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(1);
    }
    cb_received_obj.pop_front((chunks_per_row - 1) * num_candidates);
    cb_local_obj.pop_front(num_candidates);
}

}  // namespace

void kernel_main() {
    uint32_t first_unit, num_units;
    Split::core_units(first_unit, num_units);
    if (num_units == 0) {
        return;
    }
    const bool sums_row = chunks_per_row > 1 && Split::leads(first_unit);

    compute_kernel_hw_startup(cb_v, cb_v, cb_squares);
    CircularBuffer(cb_queries).wait_front(max_run);
    CircularBuffer(cb_ones).wait_front(1);

    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t num_tiles = Split::unit(u).num_tiles();
        for (uint32_t c = 0; c < num_candidates; ++c) {
            run_statistics(num_tiles, sums_row ? cb_local : cb_out);
        }
        if (sums_row) {
            sum_row();
        }
    }
}

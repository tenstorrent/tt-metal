// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weighted reduction over the candidate axis, one pass over the input:
//   dst0 += input_tile[c] * weight_col[c]   for c in [0, num_candidates)
//
// The multiply and the add are the same instruction. `init_bcast` sets up
// PACK + UNPACK + hw_configure for ELWMUL with a column broadcast; overriding
// the MATH init with acc_to_dest=1 turns every `mul_tiles_bcast_cols` into a
// MAC against dst0, which `tile_regs_acquire()` has zeroed. That is what buys
// the fusion: no intermediate CB, no second read of the input.
//
// The weight tile holds its scalar in column 0, which is what BroadcastType::COL
// reads, so a [B, C, H, 1] tile needs no pre-pass to get into this form.

#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
    constexpr uint32_t input_granularity = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);

    // runtime args
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_granules = num_candidates / input_granularity;

    CircularBuffer cb_in0_obj(cb_in0);
    CircularBuffer cb_in1_obj(cb_in1);
    CircularBuffer cb_out0_obj(cb_out0);

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_in0, cb_in1, cb_out0);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_in0, cb_in1, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_in0, cb_in1);

    // One weight set per token row, shared by the Wt output tiles in that row.
    // The reader turns the set over on the same test — `i % Wt == 0` over the
    // global tile index — so the two stay in step without a semaphore.
    uint32_t width_index = start_id % Wt;
    cb_in1_obj.wait_front(num_candidates);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        if (i != 0 && width_index == 0) {
            cb_in1_obj.pop_front(num_candidates);
            cb_in1_obj.wait_front(num_candidates);
        }

        tile_regs_acquire();
        for (uint32_t j = 0; j < num_granules; ++j) {
            cb_in0_obj.wait_front(input_granularity);
            for (uint32_t k = 0; k < input_granularity; ++k) {
                // The granule tiles the reduction exactly, so a candidate's
                // position in the weight set is its position in the stream.
                mul_tiles_bcast_cols(cb_in0, cb_in1, k, j * input_granularity + k, dst0);
            }
            cb_in0_obj.pop_front(input_granularity);
        }
        tile_regs_commit();

        cb_out0_obj.reserve_back(onetile);
        pack_reconfig_data_format(cb_out0);
        tile_regs_wait();
        pack_tile(dst0, cb_out0);
        tile_regs_release();
        cb_out0_obj.push_back(onetile);

        ++width_index;
        if (width_index == Wt) {
            width_index = 0;
        }
    }

    // Leave the CB balanced: the last row's set was waited on but never turned over.
    cb_in1_obj.pop_front(num_candidates);
}

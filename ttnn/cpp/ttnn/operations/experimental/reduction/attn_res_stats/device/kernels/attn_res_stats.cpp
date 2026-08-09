// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {

// The reduce drains `cb_tmp`, so the same buffer carries both transforms of the
// row: fill it, reduce it, fill it again. `init` re-establishes the unpack and
// pack configuration the previous reduce left changed.
template <uint32_t cb_tmp, uint32_t cb_scaler, uint32_t cb_out, uint32_t Wt, typename Init, typename TransformOne>
ALWI void reduce_transformed_row(DataflowBuffer& tmp_buf, Init init, TransformOne transform_one) {
    init();
    tmp_buf.reserve_back(Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        tile_regs_acquire();
        transform_one(wt);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_tmp, wt);
        tile_regs_release();
    }
    tmp_buf.push_back(Wt);

    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_tmp,
        cb_scaler,
        cb_out,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
}

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_v = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_q = tt::CBIndex::c_2;
    constexpr uint32_t cb_tmp = tt::CBIndex::c_6;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_v, cb_q, cb_tmp);

    DataflowBuffer v_buf(cb_v);
    DataflowBuffer q_buf(cb_q);
    DataflowBuffer tmp_buf(cb_tmp);
    DataflowBuffer scaler_buf(cb_scaler);

    // q and the reduce scaler are the same for every row this core owns; the
    // reader pushes each once and nothing pops them until the kernel is done.
    q_buf.wait_front(Wt);

    for (uint32_t row = 0; row < num_rows; ++row) {
        v_buf.wait_front(Wt);

        reduce_transformed_row<cb_tmp, cb_scaler, cb_out, Wt>(
            tmp_buf,
            [] {
                reconfig_data_format(cb_v, cb_v);
                pack_reconfig_data_format(cb_tmp);
                mul_tiles_init(cb_v, cb_v);
            },
            [](uint32_t wt) { mul_tiles(cb_v, cb_v, wt, wt, 0); });

        reduce_transformed_row<cb_tmp, cb_scaler, cb_out, Wt>(
            tmp_buf,
            [] {
                reconfig_data_format(cb_v, cb_q);
                pack_reconfig_data_format(cb_tmp);
                mul_bcast_rows_init_short(cb_v, cb_q);
            },
            [](uint32_t wt) { mul_tiles_bcast_rows(cb_v, cb_q, wt, wt, 0); });

        v_buf.pop_front(Wt);
    }

    q_buf.pop_front(Wt);
    scaler_buf.pop_front(1);
}

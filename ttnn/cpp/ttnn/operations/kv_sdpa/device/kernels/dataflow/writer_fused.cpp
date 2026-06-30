// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// FlashFused writer: generates the reduce/bcast scalars sdpa_standard() needs, then drains the
// normalized output (one seq-tile chunk, DHt tiles) to this core's Q head of the interleaved output.
void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_head = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
        /*compute_uses_reduce_tile=*/true>();
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    const uint32_t o_tb = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, o_tb);
    cb_wait_front(cb_out, DHt);
    uint32_t l1 = get_read_ptr(cb_out);
    for (uint32_t d = 0; d < DHt; ++d) {
        noc_async_write_tile(q_head * DHt + d, out_acc, l1);
        l1 += o_tb;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, DHt);
}

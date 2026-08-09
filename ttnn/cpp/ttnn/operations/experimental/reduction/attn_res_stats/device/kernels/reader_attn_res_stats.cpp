// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto v_args = TensorAccessorArgs<1>();
    constexpr auto q_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    // runtime args
    const auto v_addr = get_arg_val<uint32_t>(0);
    const auto q_addr = get_arg_val<uint32_t>(1);
    const auto num_rows = get_arg_val<uint32_t>(2);
    const auto start_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_v = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_q = 2;

    constexpr uint32_t v_tile_bytes = get_tile_size(cb_id_v);
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_id_q);

    // Both reductions are plain sums; the mean this feeds is taken against the
    // full unsharded `d` downstream, not against this rank's share.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_id_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    Noc noc;
    DataflowBuffer v_buf(cb_id_v);
    DataflowBuffer q_buf(cb_id_q);

    auto v_accessor = TensorAccessor(v_args, v_addr);
    auto q_accessor = TensorAccessor(q_args, q_addr);

    // q spans the same d as a row of v and is the same for every row this core
    // owns, so it is read once and left resident.
    q_buf.reserve_back(Wt);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        noc.async_read(q_accessor, q_buf, q_tile_bytes, {.page_id = wt}, {.offset_bytes = wt * q_tile_bytes});
    }
    noc.async_read_barrier();
    q_buf.push_back(Wt);

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t base_page = (start_row + r) * Wt;
        v_buf.reserve_back(Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc.async_read(
                v_accessor, v_buf, v_tile_bytes, {.page_id = base_page + wt}, {.offset_bytes = wt * v_tile_bytes});
        }
        noc.async_read_barrier();
        v_buf.push_back(Wt);
    }
}

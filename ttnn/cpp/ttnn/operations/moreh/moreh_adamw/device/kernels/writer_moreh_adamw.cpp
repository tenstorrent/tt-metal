// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    const auto start_id = get_arg(args::start_id);

    const auto param_addrg = TensorAccessor(tensor::param_out);
    const auto exp_avg_addrg = TensorAccessor(tensor::exp_avg_out);
    const auto exp_avg_sq_addrg = TensorAccessor(tensor::exp_avg_sq_out);

#ifdef AMSGRAD
    const auto max_exp_avg_sq_addrg = TensorAccessor(tensor::max_exp_avg_sq_out);
#endif

    Noc noc;
    DataflowBuffer dfb_param(dfb::param_out);
    DataflowBuffer dfb_exp_avg(dfb::exp_avg_out);
    DataflowBuffer dfb_exp_avg_sq(dfb::exp_avg_sq_out);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq(dfb::max_exp_avg_sq_out);
#endif

    const auto param_tile_bytes = dfb_param.get_tile_size();
    const auto exp_avg_tile_bytes = dfb_exp_avg.get_tile_size();
    const auto exp_avg_sq_tile_bytes = dfb_exp_avg_sq.get_tile_size();
#ifdef AMSGRAD
    const auto max_exp_avg_sq_tile_bytes = dfb_max_exp_avg_sq.get_tile_size();
#endif

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_param.wait_front(onetile);
        noc.async_write(dfb_param, param_addrg, param_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_param.pop_front(onetile);

        dfb_exp_avg.wait_front(onetile);
        noc.async_write(dfb_exp_avg, exp_avg_addrg, exp_avg_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_exp_avg.pop_front(onetile);

        dfb_exp_avg_sq.wait_front(onetile);
        noc.async_write(dfb_exp_avg_sq, exp_avg_sq_addrg, exp_avg_sq_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_exp_avg_sq.pop_front(onetile);

#ifdef AMSGRAD
        dfb_max_exp_avg_sq.wait_front(onetile);
        noc.async_write(
            dfb_max_exp_avg_sq, max_exp_avg_sq_addrg, max_exp_avg_sq_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_max_exp_avg_sq.pop_front(onetile);
#endif
    }
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
void kernel_main() {
    ArgFetcher arg_fetcher;
    const uint32_t src0_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t batch_num = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt_per_core = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_h = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_w = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_mask_h_w = 2;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_id_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w);
    }

    const auto s0 = TensorAccessor(src0_args, src0_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);

    constexpr uint32_t onetile = 1;
    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        uint32_t read_tile_id = start_id + wt;
        for (uint32_t b = 0; b < batch_num; ++b) {
            cb_in0.reserve_back(onetile);
            noc.async_read(s0, cb_in0, in0_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
            read_tile_id += Wt;
        }
    }
}

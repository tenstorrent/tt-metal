// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_indexer_select pool-mode writer: maps each query's selected pool positions back to entries
// through its pool list and writes the output row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"

namespace dis = deepseek_v4_prefill_indexer_select;

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(dis::pool_writer_ct_arg::CB_OUT);
    constexpr uint32_t cb_list = get_compile_time_arg_val(dis::pool_writer_ct_arg::CB_LIST);
    constexpr auto out_args = TensorAccessorArgs<dis::pool_writer_ct_arg::END, dis::pool_writer_common_arg::END>();
    const auto out =
        TensorAccessor(out_args, get_common_arg_val<uint32_t>(dis::pool_writer_common_arg::OUTPUT_ADDRESS));
    const uint32_t first_query = get_arg_val<uint32_t>(dis::pool_core_arg::FIRST_QUERY);
    const uint32_t num_queries = get_arg_val<uint32_t>(dis::pool_core_arg::NUM_QUERIES);

    Noc noc;
    experimental::CB out_cb(cb_out), list_cb(cb_list);
    for (uint32_t i = 0; i < num_queries; ++i) {
        list_cb.wait_front(1);
        out_cb.wait_front(1);
        auto* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_cb.get_read_ptr()) + dis::POOL_LIST_WORDS;
        auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_cb.get_read_ptr());
        for (uint32_t k = 0; k < dis::TOPK; ++k) {
            const uint32_t p = row[k];
            if (p != dis::SENTINEL) {
                row[k] = list[p / dis::CAND_BLOCK] * dis::CAND_BLOCK + p % dis::CAND_BLOCK;
            }
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(out_cb.get_read_ptr()),
            out,
            dis::TOPK * sizeof(uint32_t),
            {},
            {.page_id = first_query + i});
        noc.async_writes_flushed();
        out_cb.pop_front(1);
        list_cb.pop_front(1);
    }
    noc.async_write_barrier();
}

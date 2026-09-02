// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_shards = get_arg_val<uint32_t>(4);

    constexpr uint32_t dfb_id_in0 = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<2>();
    DataflowBuffer dfb_in0(dfb_id_in0);

    Noc noc;
    const auto s0 = TensorAccessor(src_args, src_addr);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_sticks;
    for (uint32_t i = start_id; i != end_id; --i) {
        for (uint32_t k = num_shards; k > 0; --k) {
            uint32_t shard_idx = k - 1;
#else
    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        for (uint32_t k = 0; k < num_shards; k++) {
            uint32_t shard_idx = k;
#endif
            uint32_t stick_index = i * num_shards + shard_idx;
            dfb_in0.reserve_back(1);
            noc.async_read(s0, dfb_in0, stick_size, {.page_id = stick_index, .offset_bytes = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0.push_back(1);
        }
    }
}

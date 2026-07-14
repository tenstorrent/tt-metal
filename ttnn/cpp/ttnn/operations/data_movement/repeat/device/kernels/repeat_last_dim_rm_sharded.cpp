// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RM last-dim sharded; noc_async_*_sharded with per-replica write offset.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

using namespace tt::data_movement::common;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t page_start = get_arg_val<uint32_t>(2);
    const uint32_t page_end = get_arg_val<uint32_t>(3);
    const uint32_t nop = get_arg_val<uint32_t>(4);

    constexpr uint32_t original_page_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t num_repeats = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3, 0>();
    constexpr auto dst_args =
        TensorAccessorArgs<src_args.next_compile_time_args_offset(), src_args.num_common_runtime_args()>();

    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(src_args, src_addr);
    const auto d = TensorAccessor(dst_args, dst_addr);

    CircularBuffer cb(cb_id_in0);
    cb.reserve_back(1);
    const uint32_t cb_slot = cb.get_write_ptr();
    cb.push_back(1);

    Noc noc;

    for (uint32_t i = page_start; i < page_end; i++) {
        noc_async_read_sharded(noc, cb_slot, s, i, 0, original_page_size_bytes);
        noc.async_read_barrier();
        for (uint32_t k = 0; k < num_repeats; k++) {
            noc_async_write_sharded(noc, cb_slot, d, i, k * original_page_size_bytes, original_page_size_bytes);
        }
        noc.async_write_barrier();
    }
}

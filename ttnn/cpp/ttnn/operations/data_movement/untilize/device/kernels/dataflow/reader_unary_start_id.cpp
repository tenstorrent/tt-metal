// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    uint32_t end_page_id = start_page_id + num_tiles;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_reserve_back(cb_id_in0, 1);

        uint64_t noc_read_addr = s.get_noc_addr(page_id);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(noc_read_addr, l1_write_addr, tile_bytes);

        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port core reader: fetches the port's first few OWN packets straight from DRAM into their staging slots (+ tag), so
// the first sends do not wait for a local row core to read its slice and push it here. The row cores skip these
// packets (the host leaves them out of their forward lists).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "agrn_common.hpp"

using namespace agrn;

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t stage_slot_bytes = get_compile_time_arg_val(1);
    constexpr auto in_args = TensorAccessorArgs<2>();

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ctr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t salt = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t list = a;  // per packet: staging slot, first input page, ntiles

    const uint32_t tag = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr_addr) + 1;
    const uint32_t stage = get_write_ptr(CB_STAGE);
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    AGRN_ZONE("pr_fetch");
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t slot = get_arg_val<uint32_t>(list + 3 * i);
        const uint32_t page = get_arg_val<uint32_t>(list + 3 * i + 1);
        const uint32_t cnt = get_arg_val<uint32_t>(list + 3 * i + 2);
        const uint32_t dst = stage + slot * stage_slot_bytes;
        for (uint32_t t = 0; t < cnt; ++t) {
            noc_async_read_tile(page + t, in, dst + t * tile_bytes);
        }
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t slot = get_arg_val<uint32_t>(list + 3 * i);
        const uint32_t cnt = get_arg_val<uint32_t>(list + 3 * i + 2);
        write_tag(stage + slot * stage_slot_bytes + cnt * tile_bytes, tag, salt);
    }
}

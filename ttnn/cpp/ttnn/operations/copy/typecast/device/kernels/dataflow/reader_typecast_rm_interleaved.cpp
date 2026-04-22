// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for RM→RM typecast with optional page padding.
//
// Reads `actual_page_bytes` from each DRAM page into a CB of `padded_page_bytes`.
// When actual_page_bytes < padded_page_bytes (row width not a multiple of 32 elements),
// the trailing bytes in the CB page are zero-filled so the unpacker does not read
// garbage beyond the last real element.

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t actual_page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t padded_page_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    constexpr bool needs_padding = (padded_page_bytes > actual_page_bytes);
    constexpr uint32_t pad_bytes = padded_page_bytes - actual_page_bytes;

    const auto s = TensorAccessor(src_args, src_addr);
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.reserve_back(1);
        const uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc.async_read(s, cb, actual_page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();

        if constexpr (needs_padding) {
            // Zero-fill the padding so the unpacker reads zeros, not garbage.
            tt_l1_ptr uint8_t* pad_ptr = reinterpret_cast<tt_l1_ptr uint8_t*>(l1_write_addr + actual_page_bytes);
            for (uint32_t j = 0; j < pad_bytes; ++j) {
                pad_ptr[j] = 0;
            }
        }

        cb.push_back(1);
    }
}

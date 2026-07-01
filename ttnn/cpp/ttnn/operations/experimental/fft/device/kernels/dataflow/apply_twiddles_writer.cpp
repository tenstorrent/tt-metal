// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_writer.cpp — BRISC1 / writer for the apply_twiddles op.
//
// For each row `r ∈ [base, base + num_rows)` this kernel waits on the
// compute kernel's CB_B_R/CB_B_I tiles, optionally truncates them to
// bf16, and writes them to the output DRAM buffers.  Page-size-safe
// addressing (InterleavedAddrGen, NOT *Fast) so the per-bank stride
// matches the allocator for ROW_MAJOR pages where page_size < tile_size.
//
// Runtime args:
//   0: out_r_addr
//   1: out_i_addr
//   2: base_row
//   3: num_rows
//   4: out_page_size_override   (bytes per output row in DRAM; 0 → ts)
//
// Compile-time args:
//   0: N1                       (output row length in elements)
//   1: OUTPUT_BF16              (0 = fp32 fast path, 1 = bf16 output)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "apply_twiddles_common.h"

void kernel_main() {
    const uint32_t out_r_addr             = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr             = get_arg_val<uint32_t>(1);
    const uint32_t base_row               = get_arg_val<uint32_t>(2);
    const uint32_t num_rows               = get_arg_val<uint32_t>(3);
    const uint32_t out_page_size_override = get_arg_val<uint32_t>(4);

    constexpr uint32_t N1          = get_compile_time_arg_val(0);
    constexpr uint32_t OUTPUT_BF16 = get_compile_time_arg_val(1);

    const uint32_t ts = get_tile_size(CB_B_R);

    uint32_t fallback_ts;
    if constexpr (OUTPUT_BF16) {
        fallback_ts = get_tile_size(CB_OUT_R_BF16);
    } else {
        fallback_ts = ts;
    }
    const uint32_t out_ps = out_page_size_override ? out_page_size_override : fallback_ts;
    const InterleavedAddrGen<true> out_r_gen = {
            .bank_base_address = out_r_addr, .page_size = out_ps};
    const InterleavedAddrGen<true> out_i_gen = {
            .bank_base_address = out_i_addr, .page_size = out_ps};

    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t row = base_row + k;

        cb_wait_front(CB_B_R, 1);
        cb_wait_front(CB_B_I, 1);

        if constexpr (OUTPUT_BF16) {
            cb_reserve_back(CB_OUT_R_BF16, 1);
            cb_reserve_back(CB_OUT_I_BF16, 1);
            const uint32_t out_r_bf16_l1 = get_write_ptr(CB_OUT_R_BF16);
            const uint32_t out_i_bf16_l1 = get_write_ptr(CB_OUT_I_BF16);

            volatile tt_l1_ptr uint32_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_B_R));
            volatile tt_l1_ptr uint32_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_B_I));
            volatile tt_l1_ptr uint16_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_i_bf16_l1);
            // Truncate fp32 → bf16 (drop low 16 bits).  Matches
            // batch_fft_writer's policy; round-to-nearest-even is a
            // future-work knob if precision becomes a concern.
            for (uint32_t i = 0; i < N1; ++i) {
                dst_r[i] = static_cast<uint16_t>(src_r[i] >> 16);
                dst_i[i] = static_cast<uint16_t>(src_i[i] >> 16);
            }
            cb_push_back(CB_OUT_R_BF16, 1);
            cb_push_back(CB_OUT_I_BF16, 1);

            noc_async_write_tile(row, out_r_gen, out_r_bf16_l1);
            noc_async_write_tile(row, out_i_gen, out_i_bf16_l1);
            noc_async_write_barrier();

            cb_pop_front(CB_OUT_R_BF16, 1);
            cb_pop_front(CB_OUT_I_BF16, 1);
        } else {
            noc_async_write_tile(row, out_r_gen, get_read_ptr(CB_B_R));
            noc_async_write_tile(row, out_i_gen, get_read_ptr(CB_B_I));
            noc_async_write_barrier();
        }

        cb_pop_front(CB_B_R, 1);
        cb_pop_front(CB_B_I, 1);
    }
}

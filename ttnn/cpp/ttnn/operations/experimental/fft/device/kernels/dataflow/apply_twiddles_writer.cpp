// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_writer.cpp — BRISC1 / writer for the apply_twiddles op.
//
// For each row `r ∈ [base, base + num_rows)` this kernel waits on the
// compute kernel's CB_B_R/CB_B_I tiles, optionally truncates them to
// bf16, and writes them to the output DRAM buffers.  Page-size-safe
// matches the allocator for ROW_MAJOR pages where page_size < tile_size.
//
// Runtime args:
//   base_row, num_rows
//
// Compile-time args:
//   n1                          (output row length in elements)
//
// Defines:
//   OUTPUT_BF16                 (set → bf16 output; unset → fp32 fast path)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "apply_twiddles_common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_row = get_arg(args::base_row);
    const uint32_t num_rows = get_arg(args::num_rows);

    constexpr uint32_t N1 = get_arg(args::n1);

    const auto out_r_gen = TensorAccessor(tensor::out_r);
    const auto out_i_gen = TensorAccessor(tensor::out_i);

    Noc noc;
    DataflowBuffer cb_b_r(dfb::b_r);
    DataflowBuffer cb_b_i(dfb::b_i);
#ifdef OUTPUT_BF16
    DataflowBuffer cb_out_r_bf16(dfb::out_r_bf16);
    DataflowBuffer cb_out_i_bf16(dfb::out_i_bf16);
#endif

    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t row = base_row + k;

        cb_b_r.wait_front(1);
        cb_b_i.wait_front(1);

#ifdef OUTPUT_BF16
        {
            cb_out_r_bf16.reserve_back(1);
            cb_out_i_bf16.reserve_back(1);
            const uint32_t out_r_bf16_l1 = cb_out_r_bf16.get_write_ptr();
            const uint32_t out_i_bf16_l1 = cb_out_i_bf16.get_write_ptr();

            volatile tt_l1_ptr uint32_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_b_r.get_read_ptr());
            volatile tt_l1_ptr uint32_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_b_i.get_read_ptr());
            volatile tt_l1_ptr uint16_t* const dst_r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const dst_i = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_i_bf16_l1);
            // Truncate fp32 → bf16 (drop low 16 bits).  Matches
            // batch_fft_writer's policy; round-to-nearest-even is a
            // future-work knob if precision becomes a concern.
            for (uint32_t i = 0; i < N1; ++i) {
                dst_r[i] = static_cast<uint16_t>(src_r[i] >> 16);
                dst_i[i] = static_cast<uint16_t>(src_i[i] >> 16);
            }
            cb_out_r_bf16.push_back(1);
            cb_out_i_bf16.push_back(1);

            noc.async_write(cb_out_r_bf16, out_r_gen, out_r_gen.get_aligned_page_size(), {}, {.page_id = row});
            noc.async_write(cb_out_i_bf16, out_i_gen, out_i_gen.get_aligned_page_size(), {}, {.page_id = row});
            noc.async_write_barrier();

            cb_out_r_bf16.pop_front(1);
            cb_out_i_bf16.pop_front(1);
        }
#else
        {
            noc.async_write(cb_b_r, out_r_gen, out_r_gen.get_aligned_page_size(), {}, {.page_id = row});
            noc.async_write(cb_b_i, out_i_gen, out_i_gen.get_aligned_page_size(), {}, {.page_id = row});
            noc.async_write_barrier();
        }
#endif

        cb_b_r.pop_front(1);
        cb_b_i.pop_front(1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_reader.cpp — BRISC0 / reader for the apply_twiddles op.
//
// For each input row index `r ∈ [base, base + num_rows)` this kernel:
//   1. DMAs input row r (real + imag) from DRAM into a fp32 L1 tile.
//      - INPUT_BF16=0 (fp32): direct DMA into CB_A_R / CB_A_I.
//      - INPUT_BF16=1 (bf16): DMA bf16 tile into CB_IN_*_BF16 then
//        in-place bit-shift-expand the first N1 elements into CB_A_R/I.
//        (NOT *Fast) so the per-bank stride is `aligned_page_size`
//        (matches the allocator), not the hardcoded tile size.  See
//        batch_fft_reader.cpp for the full rationale.
//   2. DMAs the broadcast twiddle row `(r % N2)` from the tile-padded
//      twiddle table (always fp32) into CB_T_R / CB_T_I.  The table is
//      our own buffer so its pages are kTileBytes — *Fast addressing is
//      safe here.
//
// Runtime args:
//   base_row                   (first row index this core handles)
//   num_rows                   (rows per core)
//   n2                         (twiddle modulus — row r uses tw row r%n2)
//
// Compile-time args:
//   n1                         (input row length in elements)
//
// Defines:
//   INPUT_BF16                 (set → bf16 input; unset → fp32 fast path)

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
    const uint32_t N2 = get_arg(args::n2);

    constexpr uint32_t N1 = get_arg(args::n1);

    const auto in_r_gen = TensorAccessor(tensor::in_r);
    const auto in_i_gen = TensorAccessor(tensor::in_i);

    // Twiddle table is our own tile-padded table (fp32, kTileBytes pages).
    const auto tw_r_gen = TensorAccessor(tensor::tw_r);
    const auto tw_i_gen = TensorAccessor(tensor::tw_i);

    Noc noc;
    DataflowBuffer cb_a_r(dfb::a_r);
    DataflowBuffer cb_a_i(dfb::a_i);
    DataflowBuffer cb_t_r(dfb::t_r);
    DataflowBuffer cb_t_i(dfb::t_i);
#ifdef INPUT_BF16
    DataflowBuffer cb_in_r_bf16(dfb::in_r_bf16);
    DataflowBuffer cb_in_i_bf16(dfb::in_i_bf16);
#endif

    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t row = base_row + k;
        const uint32_t tw_row = row % N2;

        // ── Reserve space for fp32 row tiles + twiddle tiles ───────────
        cb_a_r.reserve_back(1);
        cb_a_i.reserve_back(1);
        cb_t_r.reserve_back(1);
        cb_t_i.reserve_back(1);

#ifdef INPUT_BF16
        {
            // bf16 input → expand to fp32 in dfb::a_r / dfb::a_i.
            cb_in_r_bf16.reserve_back(1);
            cb_in_i_bf16.reserve_back(1);
            const uint32_t in_r_bf16_l1 = cb_in_r_bf16.get_write_ptr();
            const uint32_t in_i_bf16_l1 = cb_in_i_bf16.get_write_ptr();

            noc.async_read(in_r_gen, cb_in_r_bf16, in_r_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read(in_i_gen, cb_in_i_bf16, in_i_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read(tw_r_gen, cb_t_r, tw_r_gen.get_aligned_page_size(), {.page_id = tw_row}, {});
            noc.async_read(tw_i_gen, cb_t_i, tw_i_gen.get_aligned_page_size(), {.page_id = tw_row}, {});
            noc.async_read_barrier();

            // Expand first N1 bf16 → fp32 (shift left 16).  Slots [N1,
            // kTileElems) in CB_A_R/I are left untouched — the writer
            // only emits N1*elem_size bytes per row so garbage there
            // never reaches DRAM.
            volatile tt_l1_ptr uint16_t* const src_r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const src_i = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_i_bf16_l1);
            volatile tt_l1_ptr uint32_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_a_r.get_write_ptr());
            volatile tt_l1_ptr uint32_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_a_i.get_write_ptr());
            for (uint32_t i = 0; i < N1; ++i) {
                dst_r[i] = static_cast<uint32_t>(src_r[i]) << 16;
                dst_i[i] = static_cast<uint32_t>(src_i[i]) << 16;
            }

            cb_in_r_bf16.push_back(1);
            cb_in_i_bf16.push_back(1);

            // Pop the bf16 staging slots before the next iteration —
            // these CBs are 1-deep and have no downstream consumer, so
            // omitting the pop deadlocks the kernel on iteration 2 when
            // rows_per_core > 1 (matches batch_fft_reader's pattern).
            cb_in_r_bf16.pop_front(1);
            cb_in_i_bf16.pop_front(1);
        }
#else
        {
            noc.async_read(in_r_gen, cb_a_r, in_r_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read(in_i_gen, cb_a_i, in_i_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read(tw_r_gen, cb_t_r, tw_r_gen.get_aligned_page_size(), {.page_id = tw_row}, {});
            noc.async_read(tw_i_gen, cb_t_i, tw_i_gen.get_aligned_page_size(), {.page_id = tw_row}, {});
            noc.async_read_barrier();
        }
#endif

        cb_a_r.push_back(1);
        cb_a_i.push_back(1);
        cb_t_r.push_back(1);
        cb_t_i.push_back(1);
    }
}

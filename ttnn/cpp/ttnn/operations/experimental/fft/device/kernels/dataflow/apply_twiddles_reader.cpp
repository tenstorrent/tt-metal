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
//      - Bank-stride correctness: we use InterleavedAddrGen<true>
//        (NOT *Fast) so the per-bank stride is `aligned_page_size`
//        (matches the allocator), not the hardcoded tile size.  See
//        batch_fft_reader.cpp for the full rationale.
//   2. DMAs the broadcast twiddle row `(r % N2)` from the tile-padded
//      twiddle table (always fp32) into CB_T_R / CB_T_I.  The table is
//      our own buffer so its pages are kTileBytes — *Fast addressing is
//      safe here.
//
// Runtime args:
//   0: in_r_addr               (DRAM base address)
//   1: in_i_addr
//   2: tw_r_addr               (twiddle table, tile-padded fp32)
//   3: tw_i_addr
//   4: base_row                (first row index this core handles)
//   5: num_rows                (rows per core)
//   6: N2                      (twiddle modulus — row r uses tw row r%N2)
//   7: in_page_size_override   (bytes per input row in DRAM; 0 → ts)
//   8: in_imag_page_size_override
//
// Compile-time args:
//   0: N1                       (input row length in elements)
//   1: INPUT_BF16               (0 = fp32 fast path, 1 = bf16 input)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "apply_twiddles_common.h"

void kernel_main() {
    const uint32_t in_r_addr     = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr     = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t base_row      = get_arg_val<uint32_t>(4);
    const uint32_t num_rows      = get_arg_val<uint32_t>(5);
    const uint32_t N2            = get_arg_val<uint32_t>(6);
    const uint32_t in_page_size_override      = get_arg_val<uint32_t>(7);
    const uint32_t in_imag_page_size_override = get_arg_val<uint32_t>(8);

    constexpr uint32_t N1         = get_compile_time_arg_val(0);
    constexpr uint32_t INPUT_BF16 = get_compile_time_arg_val(1);

    const DataFormat df = get_dataformat(CB_A_R);
    const uint32_t   ts = get_tile_size(CB_A_R);

    // ── Input addressing (see batch_fft_reader for InterleavedAddrGen vs
    //    *Fast rationale: *Fast hard-codes within-bank stride to
    //    tile_size(data_format) which is wrong for ROW_MAJOR pages that
    //    are smaller than a tile). ────────────────────────────────────
    uint32_t fallback_ts;
    if constexpr (INPUT_BF16) {
        fallback_ts = get_tile_size(CB_IN_R_BF16);
    } else {
        fallback_ts = ts;
    }
    const uint32_t in_r_ps = in_page_size_override      ? in_page_size_override      : fallback_ts;
    const uint32_t in_i_ps = in_imag_page_size_override ? in_imag_page_size_override : fallback_ts;
    const InterleavedAddrGen<true> in_r_gen = {
            .bank_base_address = in_r_addr, .page_size = in_r_ps};
    const InterleavedAddrGen<true> in_i_gen = {
            .bank_base_address = in_i_addr, .page_size = in_i_ps};

    // Twiddle table is our own tile-padded buffer (fp32, kTileBytes pages).
    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t row    = base_row + k;
        const uint32_t tw_row = row % N2;

        // ── Reserve space for fp32 row tiles + twiddle tiles ───────────
        cb_reserve_back(CB_A_R, 1);
        cb_reserve_back(CB_A_I, 1);
        cb_reserve_back(CB_T_R, 1);
        cb_reserve_back(CB_T_I, 1);

        if constexpr (INPUT_BF16) {
            // bf16 input → expand to fp32 in CB_A_R/I.
            cb_reserve_back(CB_IN_R_BF16, 1);
            cb_reserve_back(CB_IN_I_BF16, 1);
            const uint32_t in_r_bf16_l1 = get_write_ptr(CB_IN_R_BF16);
            const uint32_t in_i_bf16_l1 = get_write_ptr(CB_IN_I_BF16);

            noc_async_read_tile(row, in_r_gen, in_r_bf16_l1);
            noc_async_read_tile(row, in_i_gen, in_i_bf16_l1);
            noc_async_read_tile(tw_row, tw_r_gen, get_write_ptr(CB_T_R));
            noc_async_read_tile(tw_row, tw_i_gen, get_write_ptr(CB_T_I));
            noc_async_read_barrier();

            // Expand first N1 bf16 → fp32 (shift left 16).  Slots [N1,
            // kTileElems) in CB_A_R/I are left untouched — the writer
            // only emits N1*elem_size bytes per row so garbage there
            // never reaches DRAM.
            volatile tt_l1_ptr uint16_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_i_bf16_l1);
            volatile tt_l1_ptr uint32_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_A_R));
            volatile tt_l1_ptr uint32_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_A_I));
            for (uint32_t i = 0; i < N1; ++i) {
                dst_r[i] = static_cast<uint32_t>(src_r[i]) << 16;
                dst_i[i] = static_cast<uint32_t>(src_i[i]) << 16;
            }

            cb_push_back(CB_IN_R_BF16, 1);
            cb_push_back(CB_IN_I_BF16, 1);

            // Pop the bf16 staging slots before the next iteration —
            // these CBs are 1-deep and have no downstream consumer, so
            // omitting the pop deadlocks the kernel on iteration 2 when
            // rows_per_core > 1 (matches batch_fft_reader's pattern).
            cb_pop_front(CB_IN_R_BF16, 1);
            cb_pop_front(CB_IN_I_BF16, 1);
        } else {
            noc_async_read_tile(row, in_r_gen, get_write_ptr(CB_A_R));
            noc_async_read_tile(row, in_i_gen, get_write_ptr(CB_A_I));
            noc_async_read_tile(tw_row, tw_r_gen, get_write_ptr(CB_T_R));
            noc_async_read_tile(tw_row, tw_i_gen, get_write_ptr(CB_T_I));
            noc_async_read_barrier();
        }

        cb_push_back(CB_A_R, 1);
        cb_push_back(CB_A_I, 1);
        cb_push_back(CB_T_R, 1);
        cb_push_back(CB_T_I, 1);
    }
}

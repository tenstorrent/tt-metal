// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_xl_reader.cpp — BRISC0 / reader for the apply_twiddles_xl
// op.  Same CB layout as apply_twiddles_reader.cpp (so the writer and
// compute kernels are reused verbatim), but the twiddle row is computed
// on-the-fly from a small per-(device, big_modulus, full_N) delta table:
//
//   delta[i] = exp(-2πi · i / full_N)      for i ∈ [0, big_modulus)
//   tw[r, 0] = (1, 0)
//   tw[r, k] = tw[r, k-1] · delta[r % big_modulus]      (k = 1..P-1)
//
// This lets big_modulus scale to 2^20 without blowing up the host twiddle
// table (which would otherwise be big_modulus·P × 8 bytes = up to 8 GB).
//
// Per-row DRAM cost: two full delta-table tiles (4 KB each) + one
// P-element row of the input.  We read the FULL tile per row because
// scalar (< tile-sized) NoC reads have arch-specific alignment quirks
// (16 B on WH, 32 B on some BH variants); reading a whole tile via the
// InterleavedAddrGenFast `noc_async_read_tile` API sidesteps all of
// that.  DRAM L2 caches the tile across rows, so the per-row cost is
// dominated by the L1 fill, not the DRAM fetch.
// Per-row compute cost on BRISC0: ~P fp32 multiply-adds for the
// recurrence (≈ 1 µs/row at P=1024).
//
// Runtime args:
//   0: in_r_addr               (input row DRAM base)
//   1: in_i_addr
//   2: dr_addr                 (delta table, real, tile-padded fp32)
//   3: di_addr                 (delta table, imag, tile-padded fp32)
//   4: base_row                (first row index this core handles)
//   5: num_rows                (rows per core)
//   6: big_modulus             (twiddle row modulus)
//   7: in_page_size_override   (bytes per input row in DRAM; 0 → ts)
//   8: in_imag_page_size_override
//
// Compile-time args:
//   0: P                       (input row length in elements)
//   1: INPUT_BF16              (0 = fp32 fast path, 1 = bf16 input)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "apply_twiddles_common.h"

void kernel_main() {
    const uint32_t in_r_addr     = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr     = get_arg_val<uint32_t>(1);
    const uint32_t dr_addr       = get_arg_val<uint32_t>(2);
    const uint32_t di_addr       = get_arg_val<uint32_t>(3);
    const uint32_t base_row      = get_arg_val<uint32_t>(4);
    const uint32_t num_rows      = get_arg_val<uint32_t>(5);
    const uint32_t big_modulus   = get_arg_val<uint32_t>(6);
    const uint32_t in_page_size_override      = get_arg_val<uint32_t>(7);
    const uint32_t in_imag_page_size_override = get_arg_val<uint32_t>(8);

    constexpr uint32_t P          = get_compile_time_arg_val(0);
    constexpr uint32_t INPUT_BF16 = get_compile_time_arg_val(1);

    const DataFormat df = get_dataformat(CB_A_R);
    const uint32_t   ts = get_tile_size(CB_A_R);

    // ── Input addrgen (ROW_MAJOR pages → InterleavedAddrGen, NOT *Fast).
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

    // ── Delta-table addrgen (tile-padded fp32, our own buffer).  Each
    //    tile holds kTileElems (=1024) entries; entry i lives in
    //    tile (i / kTileElems), slot (i % kTileElems).
    InterleavedAddrGenFast<true> dr_gen = {
        .bank_base_address = dr_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> di_gen = {
        .bank_base_address = di_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t row = base_row + k;

        // Reserve fp32 twiddle tiles + input tiles up front.
        cb_reserve_back(CB_A_R, 1);
        cb_reserve_back(CB_A_I, 1);
        cb_reserve_back(CB_T_R, 1);
        cb_reserve_back(CB_T_I, 1);

        const uint32_t t_r_l1 = get_write_ptr(CB_T_R);
        const uint32_t t_i_l1 = get_write_ptr(CB_T_I);

        // ── Step 1: delta lookup.  Read the full delta_tile that
        //    contains row_phase via `noc_async_read_tile` (which uses
        //    InterleavedAddrGenFast and handles all DRAM-bank
        //    alignment), then pick the right scalar slot from L1.
        //    The tile lands in CB_T_R/I temporarily; we overwrite the
        //    whole tile with the recurrence in Step 2.
        const uint32_t row_phase   = row % big_modulus;
        const uint32_t delta_tile  = row_phase / kTileElems;
        const uint32_t delta_slot  = row_phase % kTileElems;

        noc_async_read_tile(delta_tile, dr_gen, t_r_l1);
        noc_async_read_tile(delta_tile, di_gen, t_i_l1);
        noc_async_read_barrier();

        volatile tt_l1_ptr float* const tw_r =
            reinterpret_cast<volatile tt_l1_ptr float*>(t_r_l1);
        volatile tt_l1_ptr float* const tw_i =
            reinterpret_cast<volatile tt_l1_ptr float*>(t_i_l1);
        const float dr = tw_r[delta_slot];
        const float di = tw_i[delta_slot];

        // ── Step 2: build twiddle row by recurrence tw[k] = tw[k-1] · δ.
        //    Slots [0, P) hold the valid twiddle; slots [P, kTileElems)
        //    are zeroed so the SFPU cmul in the compute kernel produces
        //    no garbage in the output's padding lanes (writer only emits
        //    P elements per row to DRAM, so the padding is invisible to
        //    DRAM either way — the zero keeps it cleanly defined).
        tw_r[0] = 1.0f;
        tw_i[0] = 0.0f;
        for (uint32_t kk = 1; kk < P; ++kk) {
            const float a = tw_r[kk - 1];
            const float b = tw_i[kk - 1];
            tw_r[kk] = a * dr - b * di;
            tw_i[kk] = a * di + b * dr;
        }
        for (uint32_t kk = P; kk < kTileElems; ++kk) {
            tw_r[kk] = 0.0f;
            tw_i[kk] = 0.0f;
        }

        // ── Step 3: read input row (fp32 fast path or bf16 expand).
        if constexpr (INPUT_BF16) {
            cb_reserve_back(CB_IN_R_BF16, 1);
            cb_reserve_back(CB_IN_I_BF16, 1);
            const uint32_t in_r_bf16_l1 = get_write_ptr(CB_IN_R_BF16);
            const uint32_t in_i_bf16_l1 = get_write_ptr(CB_IN_I_BF16);

            noc_async_read_tile(row, in_r_gen, in_r_bf16_l1);
            noc_async_read_tile(row, in_i_gen, in_i_bf16_l1);
            noc_async_read_barrier();

            volatile tt_l1_ptr uint16_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_i_bf16_l1);
            volatile tt_l1_ptr uint32_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_A_R));
            volatile tt_l1_ptr uint32_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_A_I));
            for (uint32_t i = 0; i < P; ++i) {
                dst_r[i] = static_cast<uint32_t>(src_r[i]) << 16;
                dst_i[i] = static_cast<uint32_t>(src_i[i]) << 16;
            }

            cb_push_back(CB_IN_R_BF16, 1);
            cb_push_back(CB_IN_I_BF16, 1);
            cb_pop_front(CB_IN_R_BF16, 1);
            cb_pop_front(CB_IN_I_BF16, 1);
        } else {
            noc_async_read_tile(row, in_r_gen, get_write_ptr(CB_A_R));
            noc_async_read_tile(row, in_i_gen, get_write_ptr(CB_A_I));
            noc_async_read_barrier();
        }

        cb_push_back(CB_A_R, 1);
        cb_push_back(CB_A_I, 1);
        cb_push_back(CB_T_R, 1);
        cb_push_back(CB_T_I, 1);
    }
}

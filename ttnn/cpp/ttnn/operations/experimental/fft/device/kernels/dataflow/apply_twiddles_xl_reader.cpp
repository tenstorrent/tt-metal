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
// that.  DRAM L2 caches the tile across rows, so the per-row cost is
// dominated by the L1 fill, not the DRAM fetch.
// Per-row compute cost on BRISC0: ~P fp32 multiply-adds for the
// recurrence (≈ 1 µs/row at P=1024).
//
// Runtime args:
//   base_row                   (first row index this core handles)
//   num_rows                   (rows per core)
//   big_modulus                (twiddle row modulus)
//
// Compile-time args:
//   p                          (input row length in elements)
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
    const uint32_t big_modulus = get_arg(args::big_modulus);

    constexpr uint32_t P = get_arg(args::p);

    const auto in_r_gen = TensorAccessor(tensor::in_r);
    const auto in_i_gen = TensorAccessor(tensor::in_i);

    // ── Delta-table accessors (tile-padded fp32, our own table).  Each
    //    tile holds kTileElems (=1024) entries; entry i lives in
    //    tile (i / kTileElems), slot (i % kTileElems).
    const auto dr_gen = TensorAccessor(tensor::d_r);
    const auto di_gen = TensorAccessor(tensor::d_i);

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

        // Reserve fp32 twiddle tiles + input tiles up front.
        cb_a_r.reserve_back(1);
        cb_a_i.reserve_back(1);
        cb_t_r.reserve_back(1);
        cb_t_i.reserve_back(1);

        const uint32_t t_r_l1 = cb_t_r.get_write_ptr();
        const uint32_t t_i_l1 = cb_t_i.get_write_ptr();

        // ── Step 1: delta lookup.  Read the full delta_tile that
        //    contains row_phase via `noc_async_read_page` (which uses
        //    alignment), then pick the right scalar slot from L1.
        //    The tile lands in dfb::t_r / dfb::t_i temporarily; we overwrite the
        //    whole tile with the recurrence in Step 2.
        const uint32_t row_phase = row % big_modulus;
        const uint32_t delta_tile = row_phase / kTileElems;
        const uint32_t delta_slot = row_phase % kTileElems;

        noc.async_read(dr_gen, cb_t_r, dr_gen.get_aligned_page_size(), {.page_id = delta_tile}, {});
        noc.async_read(di_gen, cb_t_i, di_gen.get_aligned_page_size(), {.page_id = delta_tile}, {});
        noc.async_read_barrier();

        volatile tt_l1_ptr float* const tw_r = reinterpret_cast<volatile tt_l1_ptr float*>(t_r_l1);
        volatile tt_l1_ptr float* const tw_i = reinterpret_cast<volatile tt_l1_ptr float*>(t_i_l1);
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
#ifdef INPUT_BF16
        {
            cb_in_r_bf16.reserve_back(1);
            cb_in_i_bf16.reserve_back(1);
            const uint32_t in_r_bf16_l1 = cb_in_r_bf16.get_write_ptr();
            const uint32_t in_i_bf16_l1 = cb_in_i_bf16.get_write_ptr();

            noc.async_read(in_r_gen, cb_in_r_bf16, in_r_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read(in_i_gen, cb_in_i_bf16, in_i_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read_barrier();

            volatile tt_l1_ptr uint16_t* const src_r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const src_i = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_i_bf16_l1);
            volatile tt_l1_ptr uint32_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_a_r.get_write_ptr());
            volatile tt_l1_ptr uint32_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_a_i.get_write_ptr());
            for (uint32_t i = 0; i < P; ++i) {
                dst_r[i] = static_cast<uint32_t>(src_r[i]) << 16;
                dst_i[i] = static_cast<uint32_t>(src_i[i]) << 16;
            }

            cb_in_r_bf16.push_back(1);
            cb_in_i_bf16.push_back(1);
            cb_in_r_bf16.pop_front(1);
            cb_in_i_bf16.pop_front(1);
        }
#else
        {
            noc.async_read(in_r_gen, cb_a_r, in_r_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read(in_i_gen, cb_a_i, in_i_gen.get_aligned_page_size(), {.page_id = row}, {});
            noc.async_read_barrier();
        }
#endif

        cb_a_r.push_back(1);
        cb_a_i.push_back(1);
        cb_t_r.push_back(1);
        cb_t_i.push_back(1);
    }
}

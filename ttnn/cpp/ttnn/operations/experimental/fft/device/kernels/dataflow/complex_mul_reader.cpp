// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// complex_mul_reader.cpp — BRISC0 / reader for the complex_mul op.
// Same CB layout as apply_twiddles_(xl_)reader (so the writer and
// compute kernels are reused verbatim), but instead of building the
// "T" operand on-the-fly from a delta table, this reader loads B
// directly from DRAM as a second independent complex tensor.
//
// For each row r ∈ [base_row, base_row + num_rows):
//   - Read A_R[r, :], A_I[r, :] from DRAM into CB_A_R, CB_A_I.
//   - Read B_R[r, :], B_I[r, :] from DRAM into CB_T_R, CB_T_I.
//   - Push all four tiles for the compute kernel to consume.
//
// bf16 path: each tensor's row is read into the shared bf16 staging
// CB (CB_IN_R_BF16 / CB_IN_I_BF16), expanded to fp32 in the matching
// fp32 CB, then push/pop'd before the next read.  Reusing the staging
// tiles for both A and B keeps the total CB footprint identical to
// apply_twiddles (no extra L1 budget).
//
// Runtime args:
//   0: a_r_addr               (input A real, DRAM base)
//   1: a_i_addr               (input A imag, DRAM base)
//   2: b_r_addr               (input B real, DRAM base)
//   3: b_i_addr               (input B imag, DRAM base)
//   4: base_row               (first row index this core handles)
//   5: num_rows               (rows per core)
//   6: in_page_size_override  (bytes per input row in DRAM; 0 → ts)
//
// Compile-time args:
//   0: P                      (row length in elements, 1..1024)
//   1: INPUT_BF16             (0 = fp32 fast path, 1 = bf16 input)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "apply_twiddles_common.h"

namespace {

// Reads ONE bf16 tile from `gen` at row `row`, expands the first P
// uint16 lanes to fp32 in CB `fp32_cb`, and pushes the fp32 tile.
// The shared bf16 staging CB `bf16_cb` is push/pop'd internally so
// the SAME CB can be reused for both A and B reads.
template <uint32_t P>
FORCE_INLINE void read_bf16_row_and_expand_fp32(
    uint32_t row,
    const InterleavedAddrGen<true>& gen,
    uint32_t bf16_cb,
    uint32_t fp32_cb)
{
    cb_reserve_back(bf16_cb, 1);
    cb_reserve_back(fp32_cb, 1);
    const uint32_t bf16_l1 = get_write_ptr(bf16_cb);
    const uint32_t fp32_l1 = get_write_ptr(fp32_cb);

    noc_async_read_tile(row, gen, bf16_l1);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* const src =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bf16_l1);
    volatile tt_l1_ptr uint32_t* const dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fp32_l1);
    for (uint32_t i = 0; i < P; ++i) {
        dst[i] = static_cast<uint32_t>(src[i]) << 16;
    }

    cb_push_back(bf16_cb, 1);
    cb_push_back(fp32_cb, 1);
    cb_pop_front(bf16_cb, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t a_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t a_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t b_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t b_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t base_row  = get_arg_val<uint32_t>(4);
    const uint32_t num_rows  = get_arg_val<uint32_t>(5);
    const uint32_t in_page_size_override = get_arg_val<uint32_t>(6);

    constexpr uint32_t P          = get_compile_time_arg_val(0);
    constexpr uint32_t INPUT_BF16 = get_compile_time_arg_val(1);

    const uint32_t fp32_ts = get_tile_size(CB_A_R);
    uint32_t fallback_ts;
    if constexpr (INPUT_BF16) {
        fallback_ts = get_tile_size(CB_IN_R_BF16);
    } else {
        fallback_ts = fp32_ts;
    }
    const uint32_t in_ps = in_page_size_override ? in_page_size_override : fallback_ts;

    // All four inputs share the same shape / dtype / layout (validated
    // host-side), so they all use the same per-bank page_size.
    const InterleavedAddrGen<true> a_r_gen = {.bank_base_address = a_r_addr, .page_size = in_ps};
    const InterleavedAddrGen<true> a_i_gen = {.bank_base_address = a_i_addr, .page_size = in_ps};
    const InterleavedAddrGen<true> b_r_gen = {.bank_base_address = b_r_addr, .page_size = in_ps};
    const InterleavedAddrGen<true> b_i_gen = {.bank_base_address = b_i_addr, .page_size = in_ps};

    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t row = base_row + k;

        if constexpr (INPUT_BF16) {
            // bf16 path: stage each row through the shared bf16 CB,
            // expand to fp32 in the matching compute CB.  The 4 reads
            // happen sequentially and the bf16 staging CB is re-used.
            read_bf16_row_and_expand_fp32<P>(row, a_r_gen, CB_IN_R_BF16, CB_A_R);
            read_bf16_row_and_expand_fp32<P>(row, a_i_gen, CB_IN_I_BF16, CB_A_I);
            read_bf16_row_and_expand_fp32<P>(row, b_r_gen, CB_IN_R_BF16, CB_T_R);
            read_bf16_row_and_expand_fp32<P>(row, b_i_gen, CB_IN_I_BF16, CB_T_I);
        } else {
            // fp32 fast path: NoC reads land directly in the fp32 compute CBs.
            cb_reserve_back(CB_A_R, 1);
            cb_reserve_back(CB_A_I, 1);
            cb_reserve_back(CB_T_R, 1);
            cb_reserve_back(CB_T_I, 1);

            noc_async_read_tile(row, a_r_gen, get_write_ptr(CB_A_R));
            noc_async_read_tile(row, a_i_gen, get_write_ptr(CB_A_I));
            noc_async_read_tile(row, b_r_gen, get_write_ptr(CB_T_R));
            noc_async_read_tile(row, b_i_gen, get_write_ptr(CB_T_I));
            noc_async_read_barrier();

            cb_push_back(CB_A_R, 1);
            cb_push_back(CB_A_I, 1);
            cb_push_back(CB_T_R, 1);
            cb_push_back(CB_T_I, 1);
        }
    }
}

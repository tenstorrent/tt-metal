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
//   base_row                  (first row index this core handles)
//   num_rows                  (rows per core)
//
// Compile-time args:
//   p                         (row length in elements, 1..1024)
//
// Defines:
//   INPUT_BF16                (set → bf16 input; unset → fp32 fast path)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "apply_twiddles_common.h"
#include "experimental/kernel_args.h"

namespace {

// Reads ONE bf16 tile from `gen` at row `row`, expands the first P
// uint16 lanes to fp32 in CB `fp32_cb`, and pushes the fp32 tile.
// The shared bf16 staging CB `bf16_cb` is push/pop'd internally so
// the SAME CB can be reused for both A and B reads.
template <uint32_t P, typename AddrGen>
FORCE_INLINE void read_bf16_row_and_expand_fp32(
    uint32_t row, const AddrGen& gen, DataflowBuffer& bf16_cb, DataflowBuffer& fp32_cb, Noc& noc) {
    bf16_cb.reserve_back(1);
    fp32_cb.reserve_back(1);
    const uint32_t bf16_l1 = bf16_cb.get_write_ptr();
    const uint32_t fp32_l1 = fp32_cb.get_write_ptr();

    noc.async_read(gen, bf16_cb, gen.get_aligned_page_size(), {.page_id = row}, {});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint16_t* const src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bf16_l1);
    volatile tt_l1_ptr uint32_t* const dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fp32_l1);
    for (uint32_t i = 0; i < P; ++i) {
        dst[i] = static_cast<uint32_t>(src[i]) << 16;
    }

    bf16_cb.push_back(1);
    fp32_cb.push_back(1);
    bf16_cb.pop_front(1);
}

}  // namespace

void kernel_main() {
    const uint32_t base_row = get_arg(args::base_row);
    const uint32_t num_rows = get_arg(args::num_rows);

    constexpr uint32_t P = get_arg(args::p);

    // All four inputs share the same shape / dtype / layout (validated
    // host-side), so they all use the same per-bank page_size.
    const auto a_r_gen = TensorAccessor(tensor::a_r);
    const auto a_i_gen = TensorAccessor(tensor::a_i);
    const auto b_r_gen = TensorAccessor(tensor::b_r);
    const auto b_i_gen = TensorAccessor(tensor::b_i);

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

#ifdef INPUT_BF16
        {
            // bf16 path: stage each row through the shared bf16 CB,
            // expand to fp32 in the matching compute CB.  The 4 reads
            // happen sequentially and the bf16 staging CB is re-used.
            read_bf16_row_and_expand_fp32<P>(row, a_r_gen, cb_in_r_bf16, cb_a_r, noc);
            read_bf16_row_and_expand_fp32<P>(row, a_i_gen, cb_in_i_bf16, cb_a_i, noc);
            read_bf16_row_and_expand_fp32<P>(row, b_r_gen, cb_in_r_bf16, cb_t_r, noc);
            read_bf16_row_and_expand_fp32<P>(row, b_i_gen, cb_in_i_bf16, cb_t_i, noc);
        }
#else
        {
            // fp32 fast path: NoC reads land directly in the fp32 compute CBs.
            cb_a_r.reserve_back(1);
            cb_a_i.reserve_back(1);
            cb_t_r.reserve_back(1);
            cb_t_i.reserve_back(1);

            const uint32_t page_size = a_r_gen.get_aligned_page_size();
            noc.async_read(a_r_gen, cb_a_r, page_size, {.page_id = row}, {});
            noc.async_read(a_i_gen, cb_a_i, page_size, {.page_id = row}, {});
            noc.async_read(b_r_gen, cb_t_r, page_size, {.page_id = row}, {});
            noc.async_read(b_i_gen, cb_t_i, page_size, {.page_id = row}, {});
            noc.async_read_barrier();

            cb_a_r.push_back(1);
            cb_a_i.push_back(1);
            cb_t_r.push_back(1);
            cb_t_i.push_back(1);
        }
#endif
    }
}

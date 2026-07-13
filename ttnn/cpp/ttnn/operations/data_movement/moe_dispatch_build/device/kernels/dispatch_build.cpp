// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE capacity-dispatch build kernel (DiffusionGemma denoise).
//
// Collapses the post-cumsum tail of ``sparse_moe._build_capacity_dispatch_impl`` — the dependent
// chain gather -> sub -> mul -> add -> ge -> where -> typecast -> scatter(disp) -> scatter(comb) ->
// slice -> slice (each ttnn.scatter/gather internally round-trips TILE<->ROW_MAJOR) — into a single
// device op, cutting the serialized dispatch-build critical path in the traced denoise loop.
//
// Semantics (bit-identical to the impl kept columns [0:EC]):
//   for each token t, for each active expert slot j in [0, top_k):
//       e     = idx[t, j]                     (active expert id, in [0, E))
//       count = round(cum[t, e])              (inclusive per-expert token count, exact int in f32)
//       slot  = count - 1                      (exclusive rank == impl "pos")
//       if 0 <= slot < C:                      (else capacity overflow -> dropped, matches impl)
//           col          = e * C + slot
//           disp[t, col] = 1.0  (bf16 0x3F80)
//           comb[t, col] = vals[t, j]          (raw bf16 bits; impl scatters vals * 1.0 == vals)
//   all other columns of disp/comb are 0.
//
// Layouts (ROW_MAJOR, interleaved DRAM). Every read uses page_id = t so the TensorAccessor applies
// the buffer's true (possibly alignment-padded) page stride; the read size is the logical row size,
// so no padding is ever mis-read or over-read past the buffer:
//   cum  [1,1,S,E]  f32     : page = token t, read E*4  bytes
//   idx  [1,1,S,k]  uint32  : page = token t, read k*4  bytes
//   vals [1,1,S,k]  bf16    : page = token t, read k*2  bytes
//   disp [1,1,S,EC] bf16    : page = token t, write EC*2 bytes   (output)
//   comb [1,1,S,EC] bf16    : page = token t, write EC*2 bytes   (output)
//
// Each core owns a contiguous [start_row, end_row) band of tokens. No cross-core hazard: distinct
// experts land in distinct e*C bands, one expert never appears twice in a token's top-k, and each
// output row is owned by exactly one core.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // ----- compile-time args -----
    constexpr uint32_t top_k = get_compile_time_arg_val(0);
    constexpr uint32_t capacity = get_compile_time_arg_val(1);
    constexpr uint32_t EC = get_compile_time_arg_val(2);               // num_experts * capacity
    constexpr uint32_t cum_page_bytes = get_compile_time_arg_val(3);   // E * 4
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(4);   // top_k * 4
    constexpr uint32_t vals_page_bytes = get_compile_time_arg_val(5);  // top_k * 2
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(6);   // EC * 2

    constexpr auto cum_args = TensorAccessorArgs<7>();
    constexpr auto idx_args = TensorAccessorArgs<cum_args.next_compile_time_args_offset()>();
    constexpr auto vals_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto disp_args = TensorAccessorArgs<vals_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<disp_args.next_compile_time_args_offset()>();

    // ----- runtime args -----
    const uint32_t cum_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t vals_addr = get_arg_val<uint32_t>(2);
    const uint32_t disp_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_row = get_arg_val<uint32_t>(5);
    const uint32_t end_row = get_arg_val<uint32_t>(6);

    constexpr uint32_t onepage = 1;
    constexpr uint16_t BF16_ONE = 0x3F80;  // bf16 encoding of 1.0

    Noc noc;
    CircularBuffer cb_cum(tt::CBIndex::c_0);
    CircularBuffer cb_idx(tt::CBIndex::c_1);
    CircularBuffer cb_vals(tt::CBIndex::c_2);
    CircularBuffer cb_disp(tt::CBIndex::c_3);
    CircularBuffer cb_comb(tt::CBIndex::c_4);

    const auto s_cum = TensorAccessor(cum_args, cum_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);
    const auto s_vals = TensorAccessor(vals_args, vals_addr);
    const auto s_disp = TensorAccessor(disp_args, disp_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);

    for (uint32_t t = start_row; t < end_row; ++t) {
        // read this token's per-expert inclusive count row + its top-k experts / weights
        cb_cum.reserve_back(onepage);
        cb_idx.reserve_back(onepage);
        cb_vals.reserve_back(onepage);
        noc.async_read(s_cum, cb_cum, cum_page_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(s_idx, cb_idx, idx_page_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(s_vals, cb_vals, vals_page_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_cum.push_back(onepage);
        cb_idx.push_back(onepage);
        cb_vals.push_back(onepage);
        cb_cum.wait_front(onepage);
        cb_idx.wait_front(onepage);
        cb_vals.wait_front(onepage);
        const float* cum_ptr = reinterpret_cast<const float*>(cb_cum.get_read_ptr());
        const uint32_t* idx_ptr = reinterpret_cast<const uint32_t*>(cb_idx.get_read_ptr());
        const uint16_t* vals_ptr = reinterpret_cast<const uint16_t*>(cb_vals.get_read_ptr());  // raw bf16 bits

        // build the two output rows in L1 (zero, then scatter the kept assignments)
        cb_disp.reserve_back(onepage);
        cb_comb.reserve_back(onepage);
        volatile tt_l1_ptr uint16_t* disp_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_disp.get_write_ptr());
        volatile tt_l1_ptr uint16_t* comb_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_comb.get_write_ptr());
        for (uint32_t c = 0; c < EC; ++c) {
            disp_row[c] = 0;
            comb_row[c] = 0;
        }

        for (uint32_t j = 0; j < top_k; ++j) {
            const uint32_t e = idx_ptr[j];
            // cum is an exact non-negative integer stored as f32; +0.5 guards float repr noise.
            const int32_t count = static_cast<int32_t>(cum_ptr[e] + 0.5f);
            const int32_t slot = count - 1;
            if (slot >= 0 && static_cast<uint32_t>(slot) < capacity) {
                const uint32_t col = e * capacity + static_cast<uint32_t>(slot);
                disp_row[col] = BF16_ONE;
                comb_row[col] = vals_ptr[j];
            }
        }

        cb_cum.pop_front(onepage);
        cb_idx.pop_front(onepage);
        cb_vals.pop_front(onepage);

        cb_disp.push_back(onepage);
        cb_comb.push_back(onepage);
        cb_disp.wait_front(onepage);
        cb_comb.wait_front(onepage);
        noc.async_write(cb_disp, s_disp, out_page_bytes, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write(cb_comb, s_comb, out_page_bytes, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        cb_disp.pop_front(onepage);
        cb_comb.pop_front(onepage);
    }
}

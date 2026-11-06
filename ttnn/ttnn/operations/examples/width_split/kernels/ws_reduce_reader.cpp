// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// width_split reduce-case reader (NCRISC / NoC0).
//
// Fills the reduce scaler tile once (1/W, so the per-slice partials already carry the mean's
// 1/W factor and the combine is a plain sum), then streams this Tensix core's slice of each of
// its tile-rows into cb_in, `blk` tiles per NoC barrier.
//
// Byte-identical for both reduce variants:
//   row_split_reduce   : slice = the whole tile-row (S = Wt tiles), slice_start = 0.
//   width_split_reduce : slice = Wt / k tile-columns, slice_start = member * S.
//
// WS_ROW_MAJOR (Layout::ROW_MAJOR input, one DRAM page = one stick = one row of W bf16 datums):
// per tile-row, read this Tensix core's piece of each of the 32 sticks, i.e. 32 reads of
// slice_bytes = S * 64 B at byte offset slice_start * 64 B into the stick, one NoC barrier, then
// push S tile-sized pages of row-major data (32 rows x S tile-columns) for compute to tilize.
// A width split does not change the bytes read, only how many transfers carry them: k members
// each read 32 pieces of W * 2 / k B instead of one Tensix core reading 32 whole sticks.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr auto in_args = TensorAccessorArgs<2>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t wt = get_arg_val<uint32_t>(1);           // tiles per tile-row (whole width)
    const uint32_t slice = get_arg_val<uint32_t>(2);        // tiles of each tile-row this core owns
    const uint32_t slice_start = get_arg_val<uint32_t>(3);  // first tile-column of the slice
    const uint32_t row_start = get_arg_val<uint32_t>(4);    // first tile-row of this core's group
    const uint32_t num_rows = get_arg_val<uint32_t>(5);     // tile-rows this core's group owns
    const uint32_t blk = get_arg_val<uint32_t>(6);          // tiles per NoC barrier / CB push

    // Mean over the FULL width W = 32 * Wt: every slice applies the same 1/W, so the cross-core
    // combine is a plain sum of partials. Power-of-two widths make 1/W exact in bf16.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f / static_cast<float>(wt * 32));

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

#ifdef WS_ROW_MAJOR
    constexpr uint32_t cb_rm_in = 5;
    const uint32_t slice_bytes = get_arg_val<uint32_t>(7);   // bytes of each stick this core owns
    const uint32_t slice_offset = get_arg_val<uint32_t>(8);  // byte offset of that piece in the stick
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            const uint32_t first_stick = (row_start + r) * 32;
            cb_reserve_back(cb_rm_in, slice);
            const uint32_t l1_write_addr = get_write_ptr(cb_rm_in);
            for (uint32_t s = 0; s < 32; ++s) {
                noc_async_read(
                    in_acc.get_noc_addr(first_stick + s, slice_offset), l1_write_addr + s * slice_bytes, slice_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm_in, slice);
        }
    }
#else
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            const uint32_t first_page = (row_start + r) * wt + slice_start;
            for (uint32_t p = 0; p < slice; p += blk) {
                cb_reserve_back(cb_in, blk);
                const uint32_t l1_write_addr = get_write_ptr(cb_in);
                for (uint32_t i = 0; i < blk; ++i) {
                    noc_async_read(in_acc.get_noc_addr(first_page + p + i), l1_write_addr + i * page_bytes, page_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, blk);
            }
        }
    }
#endif
}

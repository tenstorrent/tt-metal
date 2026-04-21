// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Minimal N-input interleaved DRAM reader for kernel_lib tests.
// Compile-time arg 0 chooses how many inputs to pull (1, 2, or 3). Each input
// feeds CBs c_0 / c_1 / c_2 respectively. Runtime args are laid out as:
//   src_addr_0, src_addr_1, src_addr_2  (only the first N are consumed)
//   num_tiles
//   start_id
// followed by the TensorAccessorArgs for each enabled input.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t num_inputs = get_compile_time_arg_val(0);
    static_assert(num_inputs >= 1 && num_inputs <= 3, "reader_n_input supports 1..3 inputs");

    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_2 = tt::CBIndex::c_2;

    // TensorAccessorArgs are laid out sequentially after compile-time arg 0.
    constexpr auto s0_args = TensorAccessorArgs<1>();
    [[maybe_unused]] constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_id_0);
    [[maybe_unused]] experimental::CircularBuffer cb1(cb_id_1);
    [[maybe_unused]] experimental::CircularBuffer cb2(cb_id_2);

    const uint32_t bytes0 = get_tile_size(cb_id_0);
    [[maybe_unused]] const uint32_t bytes1 = (num_inputs >= 2) ? get_tile_size(cb_id_1) : 0;
    [[maybe_unused]] const uint32_t bytes2 = (num_inputs >= 3) ? get_tile_size(cb_id_2) : 0;

    const auto acc0 = TensorAccessor(s0_args, src0_addr, bytes0);
    [[maybe_unused]] const auto acc1 =
        (num_inputs >= 2) ? TensorAccessor(s1_args, src1_addr, bytes1) : TensorAccessor(s0_args, src0_addr, bytes0);
    [[maybe_unused]] const auto acc2 =
        (num_inputs >= 3) ? TensorAccessor(s2_args, src2_addr, bytes2) : TensorAccessor(s0_args, src0_addr, bytes0);

    constexpr uint32_t onetile = 1;
    for (uint32_t t = start_id; t < start_id + num_tiles; ++t) {
        cb0.reserve_back(onetile);
        noc.async_read(acc0, cb0, bytes0, {.page_id = t}, {.offset_bytes = 0});
        if constexpr (num_inputs >= 2) {
            cb1.reserve_back(onetile);
            noc.async_read(acc1, cb1, bytes1, {.page_id = t}, {.offset_bytes = 0});
        }
        if constexpr (num_inputs >= 3) {
            cb2.reserve_back(onetile);
            noc.async_read(acc2, cb2, bytes2, {.page_id = t}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        cb0.push_back(onetile);
        if constexpr (num_inputs >= 2) {
            cb1.push_back(onetile);
        }
        if constexpr (num_inputs >= 3) {
            cb2.push_back(onetile);
        }
    }
}

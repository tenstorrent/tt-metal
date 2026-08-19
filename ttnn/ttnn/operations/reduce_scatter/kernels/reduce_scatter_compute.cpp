// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — Phase B compute (TRISC; Phase A has no compute kernel).
//
// Local element-wise N-way tile sum via the compute-helper primitive. For each
// owned output-tile position, the reader pushes the N gathered blocks' tile at the
// slice-i source index into cb_gathered_slices (block order c = 0..N-1);
// sum_blocks waits the whole N-tile input, sums it (DST-chunked internally against
// DEST_AUTO_LIMIT, odd-N copy_tile-seeded, even-N acc_to_dest from DST's zero
// start), pops the input (pop_input = true: streaming producer/consumer CB), and
// pushes 1 reduced tile for the writer.
//
// Hardware startup stays with the kernel (binary_op_init_common — required @pre of
// sum_blocks; deliberately NOT owned by the helper).

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_gathered_slices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduced_slice = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);  // N blocks to sum

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);  // owned output-tile positions

    // Hardware startup, once per kernel, before the first sum_blocks. Both operands
    // come from cb_gathered_slices; the pack target is cb_reduced_slice.
    binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_reduced_slice);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        compute_kernel_lib::sum_blocks(
            cb_gathered_slices, cb_reduced_slice, num_devices, /*block_num_tiles=*/1, /*pop_input=*/true);
    }
}

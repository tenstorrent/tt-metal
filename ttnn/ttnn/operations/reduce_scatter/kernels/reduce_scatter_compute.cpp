// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — compute (TRISC), Phase B only (the reduce MeshProgramDescriptor
// wires this kernel; Phase A has no compute stage).
//
// Local element-wise N-way slice-tile sum. For each owned output-tile position t,
// the reader pushes the N gather blocks' slice tile into cb_gathered_slices (block
// order c=0..N-1). compute_kernel_lib::sum_blocks sums them into ONE output tile:
//     output_tile[start + t] = Σ_{c=0..N-1} gather_buffer_tile[c*P + slice_id(t)]
//
// sum_blocks owns the whole per-position protocol: wait(N)/pop(N) on the input,
// reserve(1)/push(1) on the output, the tile_regs lifecycle, DEST chunking against
// DEST_AUTO_LIMIT (= 4 under fp32_dest_acc_en — irrelevant at block_num_tiles=1),
// and the odd-N copy_tile seed. pop_input=true because cb_gathered_slices is a
// real producer/consumer CB (the llama_reduce_scatter mode, NOT all_reduce's
// resident-shell mode) — the default false would deadlock the reader on
// cb_reserve_back once the CB fills (op_design.md §R7).
//
// Hardware startup stays with the kernel (accumulate_helpers_compute.hpp banner:
// binary_op_init_common and compute_kernel_hw_startup are NOT interchangeable, and
// the helper deliberately never picks one) — the C++ sum_blocks model kernel
// (all_reduce_async/.../reduction.cpp) uses binary_op_init_common.

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_gathered_slices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_summed_slice = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);  // N blocks to sum

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);  // owned output-tile positions

    // Boot init (full hw_configure for unpack/math/pack). Both add operands come
    // from cb_gathered_slices; the pack target is cb_summed_slice.
    binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_summed_slice);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        compute_kernel_lib::sum_blocks(
            cb_gathered_slices, cb_summed_slice, num_devices, /*block_num_tiles=*/1, /*pop_input=*/true);
    }
}

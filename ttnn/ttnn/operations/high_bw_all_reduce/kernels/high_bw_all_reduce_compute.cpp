// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — reducer compute.
// reduce_block: cb_remote_partial + cb_local_input -> cb_reduced in fp32 DEST (fp32_dest_acc_en),
// packed to the wire dtype. Head chunks (per-chunk role, high_bw_all_reduce_roles.hpp) have no
// upstream partial and copy their own input; a ring reducer may mix roles chunk by chunk.
//
// Chunk-granular schedule: one helper call per block (chunk_tiles tiles). Inputs are waited
// once up front and popped once at the end of the block; the output block is reserved once and
// pushed once — matching the reader (pushes whole chunks) and the writer (waits whole chunks).
// Inside the block the helper batches tiles into DEST (block_size is clamped to the DEST
// capacity by the chain), so there is no per-tile CB handshake. InputTileMapping::Block makes the
// upfront window span the whole chunk and index a distinct tile per step.
//
// Precision path (CT arg use_sfpu_add, single source: host dtype == float32):
//  - bf16: FPU add (bf16 is lossless through SrcA/SrcB's tf32), fp32 DEST accumulate.
//  - fp32: the FPU would truncate fp32 operands to tf32 in SrcA/SrcB, so the add runs on the
//    SFPU instead — both operands are copy_tile'd into DEST (cb_remote_partial / cb_local_input
//    are tagged UnpackToDestFp32 on the host, which is legal since their only consumers here
//    are copy_tile loads) and summed by AddBinary in full fp32. The head copy is unchanged.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "high_bw_all_reduce_roles.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t cb_remote_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_local_input = get_compile_time_arg_val(1);
    constexpr uint32_t cb_reduced = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t pos = get_compile_time_arg_val(4);
    constexpr uint32_t group_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_slices = get_compile_time_arg_val(6);
    constexpr bool use_sfpu_add = get_compile_time_arg_val(7) != 0;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t reducer_idx = get_arg_val<uint32_t>(1);
    const uint32_t num_reducers = get_arg_val<uint32_t>(2);
    const ChainRoles<pos, group_size, num_slices> roles(num_reducers);

    constexpr auto block = IterationShape::tiles(chunk_tiles).block_size(chunk_tiles);
    constexpr auto in_local = input(cb_local_input, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block);
    constexpr auto in_partial =
        input(cb_remote_partial, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block);
    constexpr auto out_reduced = output(cb_reduced, ReservePolicy::Upfront, PushPolicy::AtEnd);

    compute_kernel_hw_startup(cb_remote_partial, cb_local_input, cb_reduced);
    // One zone over the whole block loop: the helpers own their CB waits, so this is occupancy
    // (wait + add), not payload — the payload is ablation-measured (changelog Perf 1).
    MaybeDeviceZoneScope("compute_main");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
#ifdef HBAR_ABLATE_COMPUTE  // perf ablation only (probes/perf1_bench.py): CB handshakes kept, no math
        const bool head = roles.is_head(reducer_idx, block_idx, num_blocks);
        cb_wait_front(cb_local_input, chunk_tiles);
        if (!head) {
            cb_wait_front(cb_remote_partial, chunk_tiles);
        }
        cb_reserve_back(cb_reduced, chunk_tiles);
        cb_push_back(cb_reduced, chunk_tiles);
        cb_pop_front(cb_local_input, chunk_tiles);
        if (!head) {
            cb_pop_front(cb_remote_partial, chunk_tiles);
        }
        continue;
#endif
        if (roles.is_head(reducer_idx, block_idx, num_blocks)) {
            copy<in_local, out_reduced>(block);
        } else if constexpr (use_sfpu_add) {
            binary_sfpu<AddBinary<>, in_partial, in_local, out_reduced>(block);
        } else {
            add<in_partial, in_local, out_reduced>(block);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — compute (TRISC). Element-wise SUM of the N devices' contributions
// to one output tile, folded into a SINGLE DEST register.
//
// ---------------------------------------------------------------------------
// RAW-API JUSTIFICATION (why no compute helper covers this step)
// ---------------------------------------------------------------------------
// 1. compute_kernel_lib::reduce (reduce_helpers_compute.hpp) CANNOT express this
//    reduction. ReduceDim offers only REDUCE_ROW / REDUCE_COL / REDUCE_SCALAR, all
//    of which reduce WITHIN a tile's 32x32 grid (documented output sizes
//    rows x batches, cols x batches, batches). all_reduce needs an element-wise sum
//    across a STACK of N tiles that PRESERVES the full 32x32 shape — N tiles in,
//    1 tile out, with no dimensional collapse. Accumulation across reduce() calls
//    does not help: each call still performs an intra-tile dimensional reduction.
// 2. eltwise_convenience.hpp / eltwise_chain.hpp (add / BinaryFpu / AddBinary) DO
//    NOT EXIST on this branch — ttnn/cpp/ttnn/kernel_lib/ tracks only
//    ccl_helpers_dataflow, dest_helpers, dfb_helpers_*, l1_helpers,
//    reduce_helpers_*, tilize_helpers and untilize_helpers. Including one would
//    fail to compile.
// 3. tilize/untilize helpers are inapplicable: input, gathered buffer and output
//    are all TILE layout, so there is no row-major boundary to cross.
// 4. compute_kernel_lib::DEST_AUTO_LIMIT (dest_helpers.hpp) IS used, as a
//    static_assert that this kernel's DEST footprint fits whatever the host
//    configured — it auto-detects 8 (bf16) / 4 (fp32 dest acc) from the JIT
//    DST_ACCUM_MODE / DST_SYNC_MODE, so the kernel can never desync from the
//    host's fp32_dest_acc_en.
//
// The chosen raw sequence mirrors the shipped, silicon-verified N-tile-sum idiom
// (all_reduce_async/reduction.cpp, reduce_scatter_minimal_async/ring_reduction.cpp,
// llama_reduce_scatter/reduction.cpp): pairwise add_tiles folded into one DEST
// accumulator, both FPU operands taken from the SAME CB at different tile indices.
//
// ---------------------------------------------------------------------------
// THE N-WAY FOLD
// ---------------------------------------------------------------------------
// acc_to_dest = false on the FIRST pair and true thereafter, which makes the fold
// independent of whether tile_regs_acquire() zeroes DEST (an assumption the shipped
// references rely on implicitly). For ODD N one tile is left over, so it is SEEDED
// into DEST with copy_tile and every pair then accumulates:
//
//   N=2  ->              (0,1) acc=false                      = t0+t1
//   N=3  -> seed t0,     (1,2) acc=true                       = t0+t1+t2
//   N=4  ->              (0,1) acc=false, (2,3) acc=true      = sum t0..t3
//   N=5  -> seed t0,     (1,2),(3,4) both acc=true            = sum t0..t4
//   N=7  -> seed t0,     (1,2),(3,4),(5,6) all acc=true       = sum t0..t6
//   N=8  ->              (0,1) acc=false, (2,3),(4,5),(6,7) acc=true
//
// Cost is ceil(N/2) FPU ops per output tile, ONE DEST register, no intermediate CB.
// This closes a real defect in the shipped C++ reference
// (all_reduce_async/reduction.cpp), whose odd-N branch is an empty
// "// TODO: Future support" that silently DROPS slice 0.
//
// N is a compile-time arg, so the odd/even split is `if constexpr` and at most two
// short add_tiles_init calls are issued per output tile (acc_to_dest is baked into
// the MOP at init time, so flipping it requires a re-init). Short inits are safe
// inside the DEST window; compute_kernel_hw_startup is NOT (it does MMIO writes)
// and is therefore called exactly once, at the top.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_shard_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_shard = get_compile_time_arg_val(3);

    static_assert(num_devices >= 2, "all_reduce needs at least 2 devices on the line");
    static_assert(
        1 <= compute_kernel_lib::DEST_AUTO_LIMIT, "the N-way fold needs one DEST register; DEST capacity is smaller");

    // Odd N leaves one tile unpaired -> seed it into DEST and accumulate every pair.
    constexpr bool seeded = (num_devices % 2u) == 1u;

    // Exactly once, before any other compute API. Both FPU operands come from
    // cb_shard_tiles, so icb0 == icb1.
    compute_kernel_hw_startup(cb_shard_tiles, cb_shard_tiles, cb_output_tiles);

    for (uint32_t p = 0; p < pages_per_shard; ++p) {
        // One block == the N devices' contribution to output tile p, in N
        // contiguous pages ordered device 0..N-1.
        cb_wait_front(cb_shard_tiles, num_devices);

        tile_regs_acquire();
        if constexpr (seeded) {
            copy_tile_init(cb_shard_tiles);
            copy_tile(cb_shard_tiles, 0, 0);  // DEST[0] = t0
            add_tiles_init(cb_shard_tiles, cb_shard_tiles, /*acc_to_dest=*/true);
            for (uint32_t d = 1; d + 1 < num_devices; d += 2) {
                add_tiles(cb_shard_tiles, cb_shard_tiles, d, d + 1, 0);  // DEST[0] += t_d + t_d+1
            }
        } else {
            add_tiles_init(cb_shard_tiles, cb_shard_tiles, /*acc_to_dest=*/false);
            add_tiles(cb_shard_tiles, cb_shard_tiles, 0, 1, 0);  // DEST[0] = t0 + t1
            if constexpr (num_devices > 2) {
                add_tiles_init(cb_shard_tiles, cb_shard_tiles, /*acc_to_dest=*/true);
                for (uint32_t d = 2; d + 1 < num_devices; d += 2) {
                    add_tiles(cb_shard_tiles, cb_shard_tiles, d, d + 1, 0);
                }
            }
        }
        tile_regs_commit();

        cb_pop_front(cb_shard_tiles, num_devices);

        cb_reserve_back(cb_output_tiles, 1);
        tile_regs_wait();
        // out_of_order_output defaults to false, so packing is sequential from 0
        // within the reserved region and output_tile_index would be IGNORED — we
        // reserve exactly one page, so pass no index.
        pack_tile(0, cb_output_tiles);
        tile_regs_release();
        cb_push_back(cb_output_tiles, 1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-capable exp(in) covering both valid-tail and fixed-size physical-tail synchronization.
//
// Each row has `Wt` valid tiles. The reader and writer exchange round_up(Wt, block_size)
// pages per row in FullBlock mode, but the chain executes only Ht*Wt tiles. ValidTiles mode
// instead exchanges only the logical remainder. Neither mode adds padding math.
//
// CT args: [Ht, Wt, block_size, synchronize_full_block].

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr bool synchronize_full_block = get_compile_time_arg_val(3) != 0;

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    constexpr auto tail_sync = synchronize_full_block ? BlockTailSync::FullBlock : BlockTailSync::ValidTiles;
    if constexpr (Ht == 1) {
        eltwise_chain(
            IterationShape::tiles(Wt).block_size(block_size, tail_sync),
            CopyTile<
                input(cb_in, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                Dst::D0>{},
            Exp<>{},
            PackTile<output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    } else {
        eltwise_chain(
            IterationShape::grid(Ht, Wt).block_size(block_size, tail_sync),
            CopyTile<
                input(cb_in, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                Dst::D0>{},
            Exp<>{},
            PackTile<output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    }
}

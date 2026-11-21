// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add compute (add_block): NEO t of C adds this cluster's pages t, t+C, ... (its slice of every STRIDED ring).
//
// One eltwise `add` helper call covers all of this NEO's blocks: the chain's unpack/math/pack init and dtype
// reconfig are paid once per kernel; each block of `block_tiles` tiles is one wait/pop of a and b, one
// reserve/push of out and one DEST section (acquire/commit/wait/release). The last block may be short
// (BlockTailSync::ValidTiles syncs the actual count, which is what the reader delivered).
// srcA <- a, srcB <- b, FPU add into DEST (fp32 under the default config), packer rounds once to out's dtype.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    // A block is one DEST section: larger and the chain would shrink its sync quantum below the
    // ring's block, misaligning the ring windows.
    static_assert(block_tiles >= 1 && block_tiles <= ckl::DEST_AUTO_LIMIT, "block_tiles must fit one DEST section");

    const uint32_t t = get_my_thread_id();
    const uint32_t neo_tiles = num_tiles / C + (t < num_tiles % C ? 1u : 0u);

    DataflowBuffer cb_input_a_tiles(dfb::cb_input_a_tiles);
    DataflowBuffer cb_input_b_tiles(dfb::cb_input_b_tiles);
    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);
    compute_kernel_hw_startup(dfb::cb_input_a_tiles, dfb::cb_input_b_tiles, dfb::cb_output_tiles);

    if (neo_tiles == 0) {
        return;
    }
    ckl::add<
        ckl::input(
            dfb::cb_input_a_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block),
        ckl::input(
            dfb::cb_input_b_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block),
        ckl::output(dfb::cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::tiles(neo_tiles).block_size(block_tiles),
        cb_input_a_tiles,
        cb_input_b_tiles,
        cb_output_tiles);
}

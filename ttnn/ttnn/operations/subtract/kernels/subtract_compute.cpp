// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// subtract compute — subtract_block: NEO t of C computes this cluster's tiles t, t+C, ... as ONE call of the
// kernel library's `sub` helper (BinaryFpu<Sub> -> PackTile), so the chain's init and format reconfig are
// paid once per kernel. Inside the call the helper takes `sync_tiles` tiles per DEST section and per ring
// wait / pop / reserve / push (PerBlockSize policies; the ragged tail syncs its valid remainder).
//
// Each NEO sees its own slice of every STRIDED ring, so this is the single-threaded kernel over this NEO's
// share; the dealing lives in the reader, the writer and the host split.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t compute_threads = get_arg(args::compute_threads);
    constexpr uint32_t sync_tiles = get_arg(args::sync_tiles);
    const uint32_t t = get_my_thread_id();
    const uint32_t neo_tiles = num_tiles / compute_threads + (t < num_tiles % compute_threads ? 1u : 0u);

    DataflowBuffer cb_minuend_tiles(dfb::cb_minuend_tiles);
    DataflowBuffer cb_subtrahend_tiles(dfb::cb_subtrahend_tiles);
    DataflowBuffer cb_difference_tiles(dfb::cb_difference_tiles);
    compute_kernel_hw_startup(dfb::cb_minuend_tiles, dfb::cb_subtrahend_tiles, dfb::cb_difference_tiles);

    if (neo_tiles == 0) {
        return;
    }
    ckl::sub<
        ckl::input(
            dfb::cb_minuend_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block),
        ckl::input(
            dfb::cb_subtrahend_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block),
        ckl::output(dfb::cb_difference_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::tiles(neo_tiles).block_size(sync_tiles),
        cb_minuend_tiles,
        cb_subtrahend_tiles,
        cb_difference_tiles);
}

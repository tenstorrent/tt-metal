// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multiply compute: FPU multiply through the kernel library's `mul` helper (srcA = a, srcB = b), product in
// DEST (fp32 when fp32_dest_acc_en), rounded once at pack into the output ring's format.
//
// flat_stream (regime 0): multiply_block — one helper call over this NEO's share of the core's run, a and b
//   streamed one front tile at a time. NEO t of C takes tiles t, t+C, ... (each NEO sees its own slice of the
//   STRIDED rings, so this is the single-threaded loop over its share).
//
// row_bcast_strips (regime 1): multiply_bcast_block — per block of the shared walker, one helper call over
//   grid(rows, cols): a streamed per tile, b indexed by column in the resident window and row-broadcast
//   (srcB row 0 to all 32 rows). The window's lifecycle is the caller's: the reader pushes nominal
//   `block_cols` windows, so wait / pop exactly `block_cols` around the call (the helper's Row mapping would
//   wait `cols`, which differs on a short block). Real unpacks of the window happen between the wait and the
//   pop, so the wait->pop TDMA hazard (DataflowBuffer.md C8) does not arise.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "multiply_block_walker.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t regime = get_arg(args::regime);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_cols = get_arg(args::block_cols);
    const uint32_t start = get_arg(args::start);
    const uint32_t count = get_arg(args::count);

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    DataflowBuffer dfb_out(dfb::out);
    compute_kernel_hw_startup(dfb::in_a, dfb::in_b, dfb::out);

    constexpr auto a_stream =
        ckl::input(dfb::in_a, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::InputTileMapping::Scalar);

    if constexpr (regime == 0) {
        const uint32_t t = get_my_thread_id();
        const uint32_t my_tiles = count / C + (t < count % C ? 1u : 0u);
        ckl::mul<
            a_stream,
            ckl::input(dfb::in_b, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::InputTileMapping::Scalar),
            ckl::output(dfb::out)>(ckl::IterationShape::tiles(my_tiles), dfb_a, dfb_b, dfb_out);
    } else {
        static_assert(C == 1, "row_bcast_strips deals whole blocks: one compute thread");
        const uint32_t G = get_arg(args::G);
        const uint32_t Ht = get_arg(args::Ht);
        const uint32_t Wt = get_arg(args::Wt);

        multiply::BlockWalker walker(start, count, G, Ht, Wt, block_cols);
        multiply::Block blk;
        while (walker.next(blk)) {
            dfb_b.wait_front(block_cols);
            ckl::mul<
                a_stream,
                ckl::input(
                    dfb::in_b,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row),
                ckl::output(dfb::out)>(ckl::IterationShape::grid(blk.rows, blk.cols), dfb_a, dfb_b, dfb_out);
            dfb_b.pop_front(block_cols);
        }
    }
}

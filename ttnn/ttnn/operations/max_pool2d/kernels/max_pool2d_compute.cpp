// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// max_pool2d compute: NEO t of C. Block operation fold_window_max (op_design.md).
//
// Each NEO sees its own slice of the STRIDED rings, so this is the single-threaded fold over this NEO's Q stream
// tiles (the dealing lives in the reader, the writer and the host split). Per output tile: DEST slot 0 takes tap 0;
// for every further tap, slot 1 takes the tap and slot 0 = max(slot 0, slot 1); slot 0 is packed once. The max
// selects an input value exactly, so DEST width and the output dtype lose nothing.
//
// Raw LLK instead of a helper (the design's "Helpers considered and rejected", op 3):
//   - eltwise_chain / binary_sfpu<BinaryMax>: not available on Quasar.
//   - eltwise_chain with a kernel-local max element: the chain owns one dst-sync window per call and has a
//     persistent DEST accumulation only for an FPU binary (chain.hpp:302-313); a K-tap fold into one held DEST
//     slot is not expressible (2K-1 compile-time elements otherwise, K up to 25).
//   - reduce<MAX>: collapses elements within tiles; this op keeps every element and folds across K tiles.
// What would close the gap: a BinaryMax chain element available on Quasar plus an SFPU persistent-DEST fold scope.
//
// Inits are paid once per kernel: copy (unpack A + datacopy), the SFPU max, the pack target. The default unpack
// mode is used (not UnpackToDest).

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t kAcc = 0;      // DEST slot of the running max
constexpr uint32_t kOperand = 1;  // DEST slot of the tap being folded in

// fold_window_max for one output tile: K taps from the front of `taps` into one tile pushed to `pooled`.
template <uint32_t K, typename Taps, typename Pooled>
FORCE_INLINE void fold_window_max(Taps& taps, Pooled& pooled) {
    tile_regs_acquire();
    taps.wait_front(1);
    copy_tile(dfb::window_taps, 0, kAcc);
    taps.pop_front(1);
    for (uint32_t k = 1; k < K; ++k) {
        taps.wait_front(1);
        copy_tile(dfb::window_taps, 0, kOperand);
        binary_max_tile(kAcc, kOperand, kAcc);
        taps.pop_front(1);
    }
    tile_regs_commit();

    pooled.reserve_back(1);
    tile_regs_wait();
    pack_tile(kAcc, dfb::pooled_tiles);
    tile_regs_release();
    pooled.push_back(1);
}

}  // namespace

void kernel_main() {
    const uint32_t T = get_arg(args::num_tiles);  // stream tiles of this core's assignment
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t K = get_arg(args::taps);
    const uint32_t Q = (T + C - 1) / C;  // the same for every NEO: padded rounds recompute a real tile

    DataflowBuffer taps(dfb::window_taps);
    DataflowBuffer pooled(dfb::pooled_tiles);

    compute_kernel_hw_startup(dfb::window_taps, dfb::pooled_tiles);
    copy_init(dfb::window_taps);
    binary_max_tile_init();
    pack_init(dfb::pooled_tiles);

    for (uint32_t q = 0; q < Q; ++q) {
        fold_window_max<K>(taps, pooled);
    }
}

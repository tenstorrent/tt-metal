// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for the WIDTH-SHARDED toy_variance path.
//
// W is split across cores, and W is the axis the variance reduces over -- so the reduction itself
// is split and has to be finished across cores. Every core runs the same two rounds; only the root
// runs the combines.
//
//   round 1   reduce<SUM,ROW> over this core's resident shard          -> dfb::partial
//             (root) add the num_cores gathered partials               -> dfb::mean_src
//             the reader broadcasts that back as dfb::mean
//   round 2   sub<COL> against the mean, square, reduce<SUM,ROW>       -> dfb::partial
//             (root) add the num_cores gathered partials, sqrt if std  -> dfb::out_tiles
//
// The AVG reduce normalizes by REDUCE_N (the FULL width), so a core's reduce already emits its share
// of the mean and
// each combine is a plain add of num_cores tiles -- no re-weighting, and no dependence on the cores
// having equal slices beyond what the host already gates.
//
// The input shard is read TWICE (once per round) and never popped: it is borrowed L1 that the op
// does not own the lifetime of, so both readers index into it in place.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// Add `num_blocks` contiguous blocks of `num_tiles` tiles waiting in `in_dfb` into a single
// `num_tiles` block pushed to `out_dfb`, then apply `post_op` to each output tile in DST before the
// pack. Contributors are paired with FPU add_tiles accumulating in DST. This is the reducer from
// ttnn/ttnn/operations/examples/tensix_all_reduce, narrowed to one DST batch because `num_tiles`
// here is Ht, which the host keeps small.
template <uint32_t in_dfb, uint32_t out_dfb, typename PostOp>
ALWI void combine_blocks(uint32_t num_blocks, uint32_t num_tiles, PostOp post_op) {
    DataflowBuffer in(in_dfb);
    DataflowBuffer out(out_dfb);

    in.wait_front(num_blocks * num_tiles);
    out.reserve_back(num_tiles);
    tile_regs_acquire();

    // Seed DST rather than accumulating into it from the start: tile_regs_acquire() does not
    // promise a zeroed DST, so the first write has to be a write. An odd block count seeds with a
    // copy; an even one seeds with the first PAIR at acc_to_dest=false, which keeps the pairing.
    uint32_t first_pair;
    if (num_blocks & 1) {
        copy_tile_to_dst_init_short(in_dfb);
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            copy_tile(in_dfb, tile, tile);
        }
        first_pair = 1;
    } else {
        add_init(in_dfb, in_dfb, false);
        for (uint32_t tile = 0; tile < num_tiles; ++tile) {
            add_tiles(in_dfb, in_dfb, tile, num_tiles + tile, tile);
        }
        first_pair = 2;
    }

    if (first_pair < num_blocks) {
        add_init(in_dfb, in_dfb, true);
        for (uint32_t block = first_pair; block < num_blocks; block += 2) {
            for (uint32_t tile = 0; tile < num_tiles; ++tile) {
                add_tiles(in_dfb, in_dfb, block * num_tiles + tile, (block + 1) * num_tiles + tile, tile);
            }
        }
    }

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        post_op(tile);
    }

    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        pack_tile(tile, out_dfb);
    }
    tile_regs_release();
    out.push_back(num_tiles);
    in.pop_front(num_blocks * num_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt_local = get_arg(args::Wt_local);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t shard_tiles = get_arg(args::shard_tiles);
    constexpr bool COMPUTE_STD_DEV = get_arg(args::compute_std_dev) != 0;
    constexpr uint32_t REDUCE_N = get_arg(args::reduce_n);  // full row width -> AVG scaler is 1/REDUCE_N
    const uint32_t is_root = get_arg(args::is_root);

    compute_kernel_hw_startup(dfb::in_shard, dfb::scaler, dfb::partial);

    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(Ht, Wt_local, /*NC=*/1);
    constexpr auto block_shape = ckl::IterationShape::of(Ht, Wt_local);

    // ---------- round 1: this core's share of the mean ----------
    // WaitUpfrontNoPop: the shard is resident and gets read again in round 2, so the reduce indexes
    // into it rather than consuming it.
    ckl::reduce<
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        dfb::in_shard,
        dfb::scaler,
        dfb::partial,
        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ckl::ReduceAlgorithm::Auto,
        ckl::ReduceWithinTile::Collapse,
        REDUCE_N>(
        reduce_shape,
        ckl::ReduceInputMemoryLayout::contiguous(),
        ckl::NoAccumulation{},
        ckl::NoOp{},
        ckl::ReduceScaler::compute_managed());

    if (is_root) {
        combine_blocks<dfb::gather_mean, dfb::mean_src>(num_cores, Ht, [](uint32_t) {});
    }

    // ---------- round 2: this core's share of the variance ----------
    // dfb::mean is the broadcast result, one tile per output row, re-read for every column (Col).
    // dfb::in_shard is the resident block, indexed by the walk and never popped (Block).
    ckl::sub<
        ckl::input(dfb::in_shard, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
        ckl::input(
            dfb::mean, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
        ckl::output(dfb::centered_sq)>(block_shape);

    ckl::square<ckl::input(dfb::centered_sq), ckl::output(dfb::centered_sq)>(block_shape);

    ckl::reduce<
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        dfb::centered_sq,
        dfb::scaler,
        dfb::partial,
        ckl::ReduceInputPolicy::WaitAndPopPerTile,
        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ckl::ReduceAlgorithm::Auto,
        ckl::ReduceWithinTile::Collapse,
        REDUCE_N>(
        reduce_shape,
        ckl::ReduceInputMemoryLayout::contiguous(),
        ckl::NoAccumulation{},
        ckl::NoOp{},
        ckl::ReduceScaler::compute_managed());

    if (is_root) {
        if constexpr (COMPUTE_STD_DEV) {
            // sqrt runs in DST on the combined variance, before the pack -- the only place it is
            // correct, since every partial is a variance contribution and not yet a variance.
            combine_blocks<dfb::gather_var, dfb::out_tiles>(num_cores, Ht, [](uint32_t dst) {
                sqrt_tile_init();
                sqrt_tile(dst);
            });
        } else {
            combine_blocks<dfb::gather_var, dfb::out_tiles>(num_cores, Ht, [](uint32_t) {});
        }
    }

    // Release what was held across the two rounds.
    DataflowBuffer dfb_in(dfb::in_shard);
    DataflowBuffer dfb_mean(dfb::mean);
    dfb_in.pop_front(shard_tiles);
    dfb_mean.pop_front(Ht);
}

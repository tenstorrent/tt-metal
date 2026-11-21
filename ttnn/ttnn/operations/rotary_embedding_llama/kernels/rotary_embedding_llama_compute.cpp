// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rotary_embedding_llama compute: NEO t of C, per block (one user's share of neo_dim_tiles head_dim tiles):
//
//   rotate_block          rotated = x @ T            (raw LLK matmul, packed to cb rotated)
//   rotate_combine_block  out = rotated * sin_row + x * cos_row   (eltwise_chain, packed into the output shard)
//
// Each NEO sees its own slice of the STRIDED rings (x, cos, sin, rotated, output: tiles t, t+C, ...) and the
// one trans_mat tile through an ALL ring, so this is the single-threaded kernel over the NEO's share.
//
// Raw-LLK substitution (rotate_block): the matmul is written with LLK calls, not with the kernel library's
// matmul helper (matmul_block_helpers.hpp), and the eltwise chain has no matmul element. The block is
// 1 x K(=1 tile) x neo_dim_tiles, so per-tile matmul_tiles inside one DEST section per `section_tiles` is
// the block's matmul.
//
// Two Quasar specifics handled here:
//   * llk_unpack_AB_matmul adds the tile index to the ring cursor (llk_unpack_AB_matmul_api.h): the k-th
//     tile of this NEO's share is at k * stride, passed explicitly.
//   * The pack target is programmed at pack_init, not by a format reconfig: pack_init before each phase whose
//     output ring differs from the previous one's.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// T is one 32x32 tile: the whole contraction of a head_dim tile is one matmul_tiles call.
constexpr uint32_t kTransMatTile = 0;

// out = rotated * sin_row (+ or -) x * cos_row over this NEO's whole share, one chain call. `Combine` is the
// SFPU DEST-DEST binary: AddBinary for the op, SubBinary for the tag_threads probe.
template <class Combine, class Rotated, class Sin, class Input, class Cos, class Output>
ALWI void rotate_combine_block(
    uint32_t neo_dim_tiles, Rotated& rotated, Sin& sin_tiles, Input& input_tiles, Cos& cos_tiles, Output& output_tiles) {
    using ckl::BinaryFpuOp;
    using ckl::BroadcastDim;
    using ckl::Dst;
    using ckl::InputTileMapping;
    using ckl::PopPolicy;
    using ckl::WaitPolicy;
    // The whole share as one block: one wait (Upfront) / pop (AtEnd) per input ring and one reserve/push on
    // the output. The chain clamps block_size * lane width (2) to DEST capacity itself.
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(neo_dim_tiles).block_size(neo_dim_tiles),
        ckl::BinaryFpu<
            BinaryFpuOp::Mul,
            ckl::input(dfb::rotated, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            ckl::input(dfb::sin_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            Dst::D0>{rotated, sin_tiles},
        ckl::BinaryFpu<
            BinaryFpuOp::Mul,
            ckl::input(dfb::input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            ckl::input(dfb::cos_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            Dst::D1>{input_tiles, cos_tiles},
        Combine{},
        ckl::PackTile<ckl::output(dfb::output_tiles, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd), Dst::D0>{
            output_tiles});
}

}  // namespace

void kernel_main() {
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t users_per_core = get_arg(args::users_per_core);  // blocks per core
    constexpr uint32_t neo_dim_tiles = get_arg(args::neo_dim_tiles);    // this NEO's share of a block
    constexpr uint32_t tag_threads = get_arg(args::tag_threads);        // test probe: odd NEOs subtract
    // One DEST slot per rotated tile held at once.
    constexpr uint32_t section_tiles = neo_dim_tiles < ckl::DEST_AUTO_LIMIT ? neo_dim_tiles : ckl::DEST_AUTO_LIMIT;
    static_assert(C >= 1 && neo_dim_tiles >= 1, "the host deals at least one tile to every NEO");

    DataflowBuffer input_tiles(dfb::input_tiles);
    DataflowBuffer cos_tiles(dfb::cos_tiles);
    DataflowBuffer sin_tiles(dfb::sin_tiles);
    DataflowBuffer trans_mat(dfb::trans_mat);
    DataflowBuffer rotated(dfb::rotated);
    DataflowBuffer output_tiles(dfb::output_tiles);

    // Spacing of this NEO's consecutive entries on the x ring.
    const uint32_t x_stride = input_tiles.get_consumer_stride_tiles();

    // Matmul maps in0 (x) -> SrcB and in1 (T) -> SrcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::input_tiles, dfb::trans_mat, dfb::rotated);
    trans_mat.wait_front(1);  // resident for every block

    for (uint32_t block = 0; block < users_per_core; ++block) {
        // ---- rotate_block: rotated[share] = x[share] @ T ----
        // (After the first block the sources and the packer were last set up by the combine chain.)
        reconfig_data_format(dfb::trans_mat, dfb::input_tiles);
        pack_reconfig_data_format(dfb::rotated);
        pack_init(dfb::rotated);
        matmul_init(dfb::input_tiles, dfb::trans_mat);

        input_tiles.wait_front(neo_dim_tiles);  // held: x is read again by the combine
        rotated.reserve_back(neo_dim_tiles);
        for (uint32_t first = 0; first < neo_dim_tiles; first += section_tiles) {
            const uint32_t n = neo_dim_tiles - first < section_tiles ? neo_dim_tiles - first : section_tiles;
            tile_regs_acquire();
            for (uint32_t k = 0; k < n; ++k) {
                matmul_tiles(dfb::input_tiles, dfb::trans_mat, (first + k) * x_stride, kTransMatTile, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < n; ++k) {
                pack_tile(k, dfb::rotated, first + k);
            }
            tile_regs_release();
        }
        rotated.push_back(neo_dim_tiles);
        mm_uninit(dfb::input_tiles, dfb::trans_mat);

        // ---- rotate_combine_block: out[share] = rotated * sin_row + x * cos_row ----
        pack_init(dfb::output_tiles);
        if constexpr (tag_threads) {
            if (get_my_thread_id() & 1u) {
                rotate_combine_block<ckl::SubBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>(
                    neo_dim_tiles, rotated, sin_tiles, input_tiles, cos_tiles, output_tiles);
                continue;
            }
        }
        rotate_combine_block<ckl::AddBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>(
            neo_dim_tiles, rotated, sin_tiles, input_tiles, cos_tiles, output_tiles);
    }
    trans_mat.pop_front(1);
}

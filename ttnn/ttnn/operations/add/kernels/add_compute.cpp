// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add compute (add_block): NEO t of C adds this cluster's out pages t, t+C, ... (its slice of every STRIDED ring).
//
// Regimes (`bcast_mode`, add_bcast_walk.hpp):
//   FLAT / LEADING  one eltwise `add` helper call covers all of this NEO's blocks (b's pages arrive remapped by
//                   the reader for LEADING, so compute is the same). Each block of `block_tiles` tiles is one
//                   wait/pop of a and b, one reserve/push of out and one DEST section; the last block may be
//                   short (BlockTailSync::ValidTiles syncs the actual count, which is what the reader delivered).
//   COL/ROW/SCALAR  FPU add with srcB broadcast (Col / Row / Scalar). b is resident: its window is waited once
//                   when the walk's run opens it and popped once after its last run. a and out sync in uniform
//                   blocks of `block_tiles` (one wait/pop of a and one reserve/push of out per block; only the
//                   cluster's last block is short), so no window ever straddles a ring's end.
//                   Each run's part of a block
//                   (cut at row ends for Row, whose b index is the column) is one helper call over a 1 x len grid
//                   in one DEST section; with every lifecycle caller-managed (WaitPolicy::None / PopPolicy::None,
//                   ReservePolicy::None / PushPolicy::None, which the helper header hands to the caller), the helper
//                   indexes a, b and out from their fronts through TileAddressing::Offset. The first piece emits
//                   the chain's one-time setup, the others reuse it (InitReconfigOwner::Caller). Plan (1, 1, 1).
//   NUMBER          (ADD_B_IS_NUMBER) a copied into DEST slot 0, the number's float32 bits filled into slot 1,
//                   SFPU add in the fp32 DEST, packed once: the number is never rounded to a's dtype (R7). The
//                   FPU binary cannot take it: srcB would hold the number at most as tf32, and the Quasar SFPU has
//                   no add-with-scalar (`add_unary_tile` is not built for ARCH_QUASAR, binop_with_scalar.h).
// srcA <- a, srcB <- b, FPU add into DEST (fp32 under the default config), packer rounds once to out's dtype.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"

#include "add_bcast_walk.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr auto a_streamed = ckl::input(
    dfb::cb_input_a_tiles, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
constexpr auto out_streamed =
    ckl::output(dfb::cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);

#ifndef ADD_B_IS_NUMBER
// b of a resident-window regime: caller-managed wait/pop, indexed from the window's front plus a runtime base.
template <ckl::BroadcastDim Bcast, ckl::InputTileMapping Mapping, ckl::DataFormatReconfig Reconfig>
constexpr auto b_resident = ckl::input(
    dfb::cb_input_b_tiles, Bcast, ckl::WaitPolicy::None, ckl::PopPolicy::None, Mapping, Reconfig, ckl::TileAddressing::Offset);

// a and out of a resident-window regime: the caller syncs whole blocks; pieces are indexed inside them.
template <ckl::DataFormatReconfig Reconfig>
constexpr auto a_in_block = ckl::input(
    dfb::cb_input_a_tiles,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Block,
    Reconfig,
    ckl::TileAddressing::Offset);
template <ckl::DataFormatReconfig Reconfig>
constexpr auto out_in_block = ckl::output(
    dfb::cb_output_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, Reconfig, ckl::TileAddressing::Offset);

// The broadcast dimension and b's tile mapping of each resident-window regime (Row: b index = base + column;
// Col, Scalar: the window's one tile). The primary template only keeps the discarded branches well-formed.
template <uint32_t Mode>
struct ResidentB {
    static constexpr auto bcast = ckl::BroadcastDim::Scalar;
    static constexpr auto mapping = ckl::InputTileMapping::Scalar;
};
template <>
struct ResidentB<add_bcast::MODE_COL> {
    static constexpr auto bcast = ckl::BroadcastDim::Col;
    static constexpr auto mapping = ckl::InputTileMapping::Scalar;
};
template <>
struct ResidentB<add_bcast::MODE_ROW> {
    static constexpr auto bcast = ckl::BroadcastDim::Row;
    static constexpr auto mapping = ckl::InputTileMapping::Row;
};

// One piece (a run's part of an a / out block, 1 x len tiles, one DEST section). The first piece of the kernel
// owns the chain's one-time setup (unpack / math broadcast init, pack init, format reconfig); every later piece
// runs the same chain, so it leaves that setup to the first (InitReconfigOwner::Caller, reconfig off, as the
// helper requires) and is pure compute. Nothing between the pieces touches that state: only ring syncs.
template <uint32_t Mode, ckl::InitReconfigOwner Owner, class InBuf, class BBuf, class OutBuf>
ALWI void add_piece(uint32_t len, uint32_t a_base, uint32_t b_base, InBuf& a, BBuf& b, OutBuf& out) {
    constexpr auto reconfig = Owner == ckl::InitReconfigOwner::Chain ? ckl::DataFormatReconfig::Enabled
                                                                     : ckl::DataFormatReconfig::Disabled;
    constexpr auto b_spec = b_resident<ResidentB<Mode>::bcast, ResidentB<Mode>::mapping, reconfig>;
    ckl::eltwise_chain<Owner>(
        ckl::IterationShape::grid(1, len).block_size(len),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Add, a_in_block<reconfig>, b_spec>{a, b, a_base, b_base},
        ckl::PackTile<out_in_block<reconfig>>{out, a_base});
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    constexpr uint32_t mode = get_arg(args::bcast_mode);
    // DEST lanes one tile of the chain occupies: the number chain holds a and the filled number.
    constexpr uint32_t lanes = mode == add_bcast::MODE_NUMBER ? 2 : 1;
    // A block is one DEST section: larger and the chain would shrink its sync quantum below the
    // ring's block, misaligning the ring windows.
    static_assert(
        block_tiles >= 1 && block_tiles * lanes <= ckl::DEST_AUTO_LIMIT, "block_tiles must fit one DEST section");

    const uint32_t t = get_my_thread_id();
    const uint32_t neo_tiles = num_tiles / C + (t < num_tiles % C ? 1u : 0u);

    DataflowBuffer cb_input_a_tiles(dfb::cb_input_a_tiles);
    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);

#ifdef ADD_B_IS_NUMBER
    static_assert(mode == add_bcast::MODE_NUMBER, "ADD_B_IS_NUMBER goes with the number regime");
    compute_kernel_hw_startup(dfb::cb_input_a_tiles, dfb::cb_output_tiles);
    if (neo_tiles == 0) {
        return;
    }
    const uint32_t number_bits = get_arg(args::number_bits);
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(neo_tiles).block_size(block_tiles),
        ckl::CopyTile<a_streamed, ckl::Dst::D0>{cb_input_a_tiles},
        ckl::FillBitcast<ckl::Dst::D1>{number_bits},
        ckl::AddBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<out_streamed, ckl::Dst::D0>{cb_output_tiles});
#else
    DataflowBuffer cb_input_b_tiles(dfb::cb_input_b_tiles);
    compute_kernel_hw_startup(dfb::cb_input_a_tiles, dfb::cb_input_b_tiles, dfb::cb_output_tiles);
    if (neo_tiles == 0) {
        return;
    }
    if constexpr (mode == add_bcast::MODE_FLAT || mode == add_bcast::MODE_LEADING) {
        constexpr auto b_streamed = ckl::input(
            dfb::cb_input_b_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block);
        ckl::add<a_streamed, b_streamed, out_streamed>(
            ckl::IterationShape::tiles(neo_tiles).block_size(block_tiles),
            cb_input_a_tiles,
            cb_input_b_tiles,
            cb_output_tiles);
    } else {
        const uint32_t start_id = get_arg(args::start_id);
        constexpr bool uniform = get_arg(args::uniform_windows) != 0;
        const add_bcast::Geometry geom{
            get_arg(args::Ht), get_arg(args::Wt), get_arg(args::n_inner), get_arg(args::s_outer), get_arg(args::s_inner)};
        bool first_piece = true;
        const uint32_t end = start_id + num_tiles;
        uint32_t blk_tiles = 0;  // tiles of the current a / out block
        uint32_t blk_used = 0;   // of them, added so far
        add_bcast::for_each_run<mode>(geom, start_id, num_tiles, uniform, [&](const add_bcast::Run& run) {
            if (run.open) {
                cb_input_b_tiles.wait_front(run.win_tiles);
            }
            uint32_t p = run.first_page;
            const uint32_t run_end = run.first_page + run.len;
            while (p < run_end) {
                if (blk_used == blk_tiles) {
                    blk_tiles = add_bcast::min_u32(block_tiles, end - p);
                    blk_used = 0;
                    cb_input_a_tiles.wait_front(blk_tiles);
                    cb_output_tiles.reserve_back(blk_tiles);
                }
                uint32_t len = add_bcast::min_u32(run_end - p, blk_tiles - blk_used);
                uint32_t b_base = 0;
                if constexpr (mode == add_bcast::MODE_ROW) {
                    len = add_bcast::min_u32(len, geom.Wt - p % geom.Wt);  // Row indexes b by column: stay in one row
                    b_base = p % geom.Wt - run.lo;
                }
                if (first_piece) {
                    add_piece<mode, ckl::InitReconfigOwner::Chain>(
                        len, blk_used, b_base, cb_input_a_tiles, cb_input_b_tiles, cb_output_tiles);
                    first_piece = false;
                } else {
                    add_piece<mode, ckl::InitReconfigOwner::Caller>(
                        len, blk_used, b_base, cb_input_a_tiles, cb_input_b_tiles, cb_output_tiles);
                }
                blk_used += len;
                p += len;
                if (blk_used == blk_tiles) {
                    cb_input_a_tiles.pop_front(blk_tiles);
                    cb_output_tiles.push_back(blk_tiles);
                }
            }
            if (run.close) {
                cb_input_b_tiles.pop_front(run.win_tiles);
            }
        });
    }
#endif
}

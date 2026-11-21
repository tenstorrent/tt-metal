// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multiply compute. Without an activation (CT `a_activation` = 0): FPU multiply through the kernel library's
// chain (srcA = a, srcB = b), product in DEST (fp32 when fp32_dest_acc_en), rounded once at pack into the
// output ring's format.
//
// With the SiLU on a (`a_activation` = 1): silu(a) * b in the same pass over the tile, on the vector unit:
//   CopyTile a -> D0, b -> D1 (CopyTile, or UnaryBcast<Row> for a row-broadcast b), silu_tile(D0),
//   MulBinary(D0, D1) -> D0, pack.
// Why the LLK's SFPU SiLU (silu_tile) and not sigmoid_tile + a multiply: on Quasar silu_tile is one SFPU
// pass that keeps x in a register, computes sigmoid(x) (SFPNONLINEAR exp + recip) and multiplies in
// registers, so it needs neither a second DEST slot for x nor a second unpack of a. The kernel library's
// chain has no SiLU element, so `Silu` below is a chain element written on the library's own UnaryOp base
// (the extension pattern chain.inl documents), not raw LLK around the chain. The product with b is the
// SFPU MulBinary: the FPU's DEST-reuse multiply (DestReuseBinary) takes no srcB broadcast, and it would round
// silu(a) to srcA's 19 bits; the SFPU product stays in 32-bit DEST.
//
// flat_stream (regime 0), and sharded_local (regime 2) with b of a's shape: one chain call over this NEO's
//   share of the core's run, a and b streamed one front tile at a time. NEO t of C takes tiles t, t+C, ...
//
// row_bcast_strips (regime 1): per block of the shared walker, one chain call over grid(rows, cols): a
//   streamed per tile, b indexed by column in the resident window and row-broadcast (srcB row 0 to all 32
//   rows). The window's lifecycle is the caller's: the reader pushes nominal `block_cols` windows, so wait /
//   pop exactly `block_cols` around the call (the helper's Row mapping would wait `cols`, which differs on a
//   short block). Real unpacks of the window happen between the wait and the pop, so the wait->pop TDMA
//   hazard (DataflowBuffer.md C8) does not arise.
//
// sharded_local (regime 2) with b of size 1 along H: the shard is the per-core block. b's whole resident
//   shard (one tile row of `block_cols` tiles per b image) is waited once and popped at the end; one chain
//   call per output image over grid(Ht, block_cols), b's row indexed at the tile offset of that image's b row.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "multiply_block_walker.hpp"

namespace ckl = compute_kernel_lib;

namespace multiply {

// SiLU on a DEST slot: the LLK's silu_tile as a chain element (see the file comment).
template <ckl::Dst Slot = ckl::Dst::D0>
struct Silu : ckl::UnaryOp<Silu<Slot>, Slot> {
    static ALWI void init() { silu_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { silu_tile(ckl::to_u32(Slot) + slot_offset); }
};

// A chain element bound to its buffer, with the runtime tile base only when the spec addresses by offset.
template <class Element, ckl::InputSpec Spec, class Buffer>
ALWI Element bind(Buffer& buffer, uint32_t base) {
    if constexpr (Spec.addressing == ckl::TileAddressing::Direct) {
        return Element{buffer};
    } else {
        return Element{buffer, base};
    }
}

// The block operation of every regime: out = act(a) * b over `shape`, b's tiles from `b_base` when its spec
// addresses by offset.
template <
    uint32_t act,
    ckl::InputSpec A,
    ckl::BroadcastInputSpec B,
    ckl::OutputSpec Out,
    class BufferA,
    class BufferB,
    class BufferOut>
ALWI void product(ckl::IterationShape shape, BufferA& a, BufferB& b, BufferOut& out, uint32_t b_base = 0) {
    using ckl::Dst;
    if constexpr (act == 0) {
        using Mul = ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, A, B, Dst::D0, Out.dest_accumulation>;
        if constexpr (B.input_spec.addressing == ckl::TileAddressing::Direct) {
            ckl::eltwise_chain(shape, Mul{a, b}, ckl::PackTile<Out>{out});
        } else {
            ckl::eltwise_chain(shape, Mul{a, b, 0u, b_base}, ckl::PackTile<Out>{out});
        }
    } else {
        static_assert(act == 1, "a_activation: 0 (none) or 1 (silu)");
        using LoadB = std::conditional_t<
            B.broadcast == ckl::BroadcastDim::None,
            ckl::CopyTile<B.input_spec, Dst::D1>,
            ckl::UnaryBcast<B.broadcast, B.input_spec, Dst::D1>>;
        ckl::eltwise_chain(
            shape,
            ckl::CopyTile<A, Dst::D0>{a},
            bind<LoadB, B.input_spec>(b, b_base),
            Silu<Dst::D0>{},
            ckl::MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            ckl::PackTile<Out>{out});
    }
}

}  // namespace multiply

void kernel_main() {
    constexpr uint32_t regime = get_arg(args::regime);
    constexpr uint32_t act = get_arg(args::a_activation);
    constexpr uint32_t b_bcast_row = get_arg(args::b_bcast_row);
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
    constexpr auto out_stream = ckl::output(dfb::out);

    if constexpr (regime == 0 || (regime == 2 && b_bcast_row == 0)) {
        const uint32_t t = get_my_thread_id();
        const uint32_t my_tiles = count / C + (t < count % C ? 1u : 0u);
        multiply::product<
            act,
            a_stream,
            ckl::input(dfb::in_b, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::InputTileMapping::Scalar),
            out_stream>(ckl::IterationShape::tiles(my_tiles), dfb_a, dfb_b, dfb_out);
    } else if constexpr (regime == 1) {
        static_assert(regime != 1 || C == 1, "row_bcast_strips deals whole blocks: one compute thread");
        const uint32_t G = get_arg(args::G);
        const uint32_t Ht = get_arg(args::Ht);
        const uint32_t Wt = get_arg(args::Wt);

        multiply::BlockWalker walker(start, count, G, Ht, Wt, block_cols);
        multiply::Block blk;
        while (walker.next(blk)) {
            dfb_b.wait_front(block_cols);
            multiply::product<
                act,
                a_stream,
                ckl::input(
                    dfb::in_b,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row),
                out_stream>(ckl::IterationShape::grid(blk.rows, blk.cols), dfb_a, dfb_b, dfb_out);
            dfb_b.pop_front(block_cols);
        }
    } else {
        static_assert(regime != 2 || C == 1, "sharded_local walks the resident shard: one compute thread");
        // a's shard: n_img images x Ht tile rows x block_cols tile columns; b's shard: one tile row per b image.
        const uint32_t Ht = get_arg(args::Ht);
        const uint32_t o1 = get_arg(args::o1);
        const uint32_t sb0 = get_arg(args::sb0);
        const uint32_t sb1 = get_arg(args::sb1);
        const uint32_t b_tiles = get_arg(args::b_tiles);
        const uint32_t n_img = count / (Ht * block_cols);

        dfb_b.wait_front(b_tiles);
        for (uint32_t n = 0; n < n_img; ++n) {
            const uint32_t d0 = n / o1;
            const uint32_t d1 = n - d0 * o1;
            multiply::product<
                act,
                a_stream,
                ckl::input(
                    dfb::in_b,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Offset),
                out_stream>(
                ckl::IterationShape::grid(Ht, block_cols), dfb_a, dfb_b, dfb_out, (d0 * sb0 + d1 * sb1) * block_cols);
        }
        dfb_b.pop_front(b_tiles);
    }
}

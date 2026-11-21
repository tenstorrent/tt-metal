// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// sigmoid compute (sigmoid_block): NEO t of C computes this cluster's pages t, t+C, ... (its slice of the
// STRIDED rings) as ONE eltwise_chain call:
//
//     CopyTile(cb_input_tiles) -> SigmoidAccurate (in DEST) -> PackTile(cb_output_tiles)
//
// in sync blocks of `block_tiles` pages (one DEST window: one acquire/commit/wait/release and one ring
// wait/pop/reserve/push per block, the last block short). The chain owns the DEST window, the ring lifecycle,
// the data-format reconfig and the per-element init (paid once per call: the chain is uniform).
//
// Helper deviation, recorded as the design requires (op_design.md, API Mapping): the sigmoid element is an
// op-local chain element with a raw SFPI body instead of the library's `Sigmoid<>`. On Quasar `Sigmoid<>` ->
// `sigmoid_tile` -> `calculate_sigmoid` is SFPNONLINEAR EXP + SFPNONLINEAR RECIP, both LUT approximations
// (~7-bit reciprocal mantissa, ckernel_sfpu_recip.h), which cannot meet the specification's 1-2 bf16 ULP
// numeric entry. A helper-only composition
// (Negative -> Exp -> +1 -> Recip) has no "+1" element here: add_unary_tile is compiled out on Quasar
// (binop_with_scalar.h). The element below is still a chain element, so the kernel stays one helper call.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// fp32-accurate sigmoid over one DEST tile (ITERATIONS vector rows of a face, the params wrapper walks the
// faces). `_sfpu_sigmoid_<fp32>` = 1 / (1 + exp(-x)) with the Cody-Waite + degree-6 polynomial exp and a
// LUT reciprocal refined by two Newton steps (reads vConstFloatPrgm0 = 2.0, programmed by sigmoid_init).
//
// x is clamped to [-88, 88] first: the exp's range reduction rounds through 2^23 + 2^22 and is valid only for
// |x / ln2| < 2^22; sigmoid is saturated in fp32 well inside the clamp (1 - e^-88 rounds to 1, e^-88 lies in
// the specification's tiny-reference band).
//
// Negate (test probe only): computes sigmoid(-x), so a test can tell which NEO computed a page.
template <bool is_fp32_dest_acc_en, bool Negate, int ITERATIONS>
inline void calculate_sigmoid_accurate() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        if constexpr (Negate) {
            x = -x;
        }
        x = sfpi::clamp(x, -88.0f, 88.0f);
        sfpi::dst_reg[0] = _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace ckl = compute_kernel_lib;

// The chain element: a DEST-only unary op on Slot (the UnaryOp CRTP base supplies exec / lane_width).
template <bool Negate, ckl::Dst Slot = ckl::Dst::D0>
struct SigmoidAccurate : ckl::UnaryOp<SigmoidAccurate<Negate, Slot>, Slot> {
    static ALWI void init() {
        // Not sigmoid_tile_init(): on Quasar that one leaves vConstFloatPrgm0 unprogrammed (LUT path).
        MATH(SFPU_UNARY_INIT_FN(sigmoid, ckernel::sfpu::sigmoid_init, (false)));
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sigmoid_accurate,
            (DST_ACCUM_MODE, Negate, 8),
            ckl::to_u32(Slot) + slot_offset,
            VectorMode::RC));
    }
};

template <bool Negate, class InputBuffer, class OutputBuffer>
ALWI void sigmoid_block(
    uint32_t neo_tiles, uint32_t block_tiles, InputBuffer& cb_input_tiles, OutputBuffer& cb_output_tiles) {
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(neo_tiles).block_size(block_tiles),
        ckl::CopyTile<ckl::input(
            dfb::cb_input_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block)>{cb_input_tiles},
        SigmoidAccurate<Negate>{},
        ckl::PackTile<ckl::output(dfb::cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{
            cb_output_tiles});
}

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);  // this cluster's run
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    constexpr uint32_t probe_tag_neo = get_arg(args::probe_tag_neo);
    const uint32_t t = get_my_thread_id();
    // The STRIDED ring deals page k to NEO k mod C: this NEO's share of the run.
    const uint32_t neo_tiles = num_tiles / C + (t < num_tiles % C ? 1u : 0u);

    DataflowBuffer cb_input_tiles(dfb::cb_input_tiles);
    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);
    compute_kernel_hw_startup(dfb::cb_input_tiles, dfb::cb_output_tiles);

    if constexpr (probe_tag_neo) {
        if (t & 1u) {
            sigmoid_block<true>(neo_tiles, block_tiles, cb_input_tiles, cb_output_tiles);
            return;
        }
    }
    sigmoid_block<false>(neo_tiles, block_tiles, cb_input_tiles, cb_output_tiles);
}

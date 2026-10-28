// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice compute — HELPER-COMPATIBLE candidate (perf experiment, not the op).
// Identical to the op's kernel (one eltwise_chain: CopyTile -> Clamp -> Square -> PackTile, FullBlock,
// block_tiles DEST lanes) except Clamp is replaced by a kernel-local chain element `ClampHoisted`.
//
// Why: the library calculate_clamp (ckernel_sfpu_clamp.h) keeps both bounds in ONE parameter register
// (vConstIntPrgm0 / LREG12) and re-writes it twice per SFPU row (SFPLOADI x2 + SFPCONFIG each), with a
// LOAD/SWAP/STORE per bound — ~14 cycles/row, 453 cycles/tile measured. ClampHoisted writes lo->Prgm0,
// hi->Prgm1 once per tile and runs LOAD, SWAP(max lo), SWAP(min hi), STORE per row (~185 cycles/tile).
// Same SFPSWAP min/max ops => bit-identical output. Square's init (re-issued by the chain every block,
// since Clamp/Square are non-uniform SFPU types) restores Prgm0/Prgm1 for its RNE before Square runs.
//
// CT args: [cb_input_tiles, cb_output_tiles, block_tiles]
// RT args: [core_tiles, min_bits (fp32), max_bits (fp32)]

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TRISC_MATH
namespace clamp_hoisted_sfpu {
inline void clamp_face_rows() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, 9);                               // max(x, lo)
        TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // min(., hi)
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}
inline void clamp_tile_hoisted(uint32_t idst, uint32_t lo, uint32_t hi) {
    sfpi::vConstIntPrgm0 = lo;
    sfpi::vConstIntPrgm1 = hi;
    _llk_math_eltwise_unary_sfpu_params_(clamp_face_rows, idst, VectorMode::RC);
}
}  // namespace clamp_hoisted_sfpu
#endif

template <compute_kernel_lib::Dst Slot>
struct ClampHoisted : compute_kernel_lib::UnaryOp<ClampHoisted<Slot>, Slot> {
    uint32_t min_param;
    uint32_t max_param;
    constexpr ClampHoisted(uint32_t lo, uint32_t hi) noexcept : min_param(lo), max_param(hi) {}
    static ALWI void init() { clamp_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((clamp_hoisted_sfpu::clamp_tile_hoisted(
            compute_kernel_lib::to_u32(Slot) + slot_offset, min_param, max_param)));
    }
};

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    static_assert(
        block_tiles >= 1 && block_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "FullBlock block must fit DEST");

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);
    const uint32_t min_bits = get_arg_val<uint32_t>(1);
    const uint32_t max_bits = get_arg_val<uint32_t>(2);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    using namespace compute_kernel_lib;
    MaybeDeviceZoneScope("compute_chain");
    eltwise_chain(
        IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
        CopyTile<
            input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
            Dst::D0>{input_tiles},
        ClampHoisted<Dst::D0>{min_bits, max_bits},
        Square<Dst::D0>{},
        PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
}

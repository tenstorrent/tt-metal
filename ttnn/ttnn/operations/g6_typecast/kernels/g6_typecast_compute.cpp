// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast compute — block operation `convert_block` (op_design.md, Block schedule).
//
// Realization note (deviation from the design's API Mapping row): the design names one
// `typecast<PerBlockSize...>(tiles(core_tiles).block_size(block_tiles, FullBlock))` call, but the
// eltwise chain caps a PerBlockSize/FullBlock block at the DEST capacity (chain_max_block_v: 4 tiles
// with fp32 DEST, 8 with 16-bit DEST — chain.inl ASSERT(!synchronize_full_blocks || block_size <=
// max_block)), so it cannot express a CB handshake of `block_tiles` (= the whole per-core range).
// The same block operation is therefore one chain call per block with an Upfront wait / AtEnd pop
// on cb_input_tiles and an Upfront reserve / AtEnd push on cb_output_tiles over the block's valid
// extent: ONE CB handshake per block per CB, DEST sub-blocking inside the chain (block_size is
// clamped to DEST capacity by the chain). The chain elements are exactly the ones
// compute_kernel_lib::typecast() builds (CopyTile -> Typecast<in_df,out_df> -> PackTile, formats
// derived from the CB formats); eltwise_chain is called directly only to reach InitReconfigOwner:
// the caller emits the element inits once after hw_startup and every block runs InitReconfigOwner::Caller
// (Perf 1), so setup is paid once per kernel as the design intends.
// Conversion arms (typecast.h): bf16->fp32 and bf16->bf8b in the packer; bf16->int32 SFPU truncation;
// fp32->bf16 SFPU RNE (ties-to-even); uint16->int32 SFPU zero-extend.
// int32->bf16: the direct Int32->Float16_b LLK arm rounds fp32->bf16 ties half-away (SFP_STOCH_RND), so it
// gets e.g. 33685505 -> 33816576 wrong; torch rounds through float32 with RNE twice. Perf 1 replaced the exact
// but non-hoistable Typecast<Int32,Float32> -> Typecast<Float32,Float16_b> pair (re-init per tile) with ONE
// custom element, Int32ToBf16Rne (g6_typecast_sfpu_elements.hpp, raw-LLK justification there).

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "g6_typecast_zones.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "g6_typecast_sfpu_elements.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    // Faces per CB tile: 4 = whole 32x32 tiles; 2 / 1 = Perf 1 sub-tile split (the CBs are declared with a 32x16 /
    // 16x16 tiny tile, so hw_startup programs unpack/datacopy/pack for NF faces and the SFPU walks NF faces).
    constexpr uint32_t NF = get_compile_time_arg_val(3);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;
    constexpr auto in_df = dfb_l1_format<cb_input_tiles>();
    constexpr auto out_df = dfb_l1_format<cb_output_tiles>();
    // Every block: the caller emitted the inits, hw_startup the data formats (Caller-owned, reconfig disabled).
    constexpr InputSpec in_spec_hoisted = input(
        cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block, DataFormatReconfig::Disabled);
    constexpr OutputSpec out_spec_hoisted =
        output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled);

    DataflowBuffer dfb_in(cb_input_tiles);
    DataflowBuffer dfb_out(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    // Arm selection: every arm is ONE chain element. int32 -> bf16 uses the custom single-init RNE element
    // (g6_typecast_sfpu_elements.hpp, raw-LLK justification there); every other arm is TypecastNF<in, out, NF>
    // (= Typecast for whole tiles).
    constexpr uint32_t df_int32 = static_cast<uint32_t>(DataFormat::Int32);
    constexpr uint32_t df_bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
    constexpr bool int32_to_bf16 = in_df == df_int32 && out_df == df_bf16;
    auto element = []() {
        if constexpr (int32_to_bf16) {
            return Int32ToBf16Rne<NF>{};
        } else {
            return TypecastNF<in_df, out_df, NF>{};
        }
    };
    // Perf 1 (lean_compute_startup): the element inits are emitted ONCE by the caller right after hw_startup,
    // and EVERY block, the first included, runs Caller-owned with data-format reconfig disabled:
    // compute_kernel_hw_startup already programmed unpack/math/pack for exactly these two CBs, so the
    // InitReconfigOwner::Chain first call only re-emitted the same reconfig (Q2198 with the lean dataflow:
    // 903 -> 884 ns; pack thread busy 359 -> ~218 ns).
    copy_init(cb_input_tiles);
    decltype(element())::init();

    G6_ZONE("CONVERT");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t block_extent = (core_tiles - done < block_tiles) ? (core_tiles - done) : block_tiles;
        const auto shape = IterationShape::tiles(block_extent).block_size(block_extent);
        eltwise_chain<InitReconfigOwner::Caller>(
            shape, CopyTile<in_spec_hoisted>{dfb_in}, element(), PackTile<out_spec_hoisted>{dfb_out});
    }
}

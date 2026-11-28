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
// the first block's call emits init + format reconfig, every later block reuses it
// (InitReconfigOwner::Caller), so setup is paid once per kernel as the design intends.
// Conversion arms (typecast.h): bf16->fp32 and bf16->bf8b in the packer; bf16->int32 SFPU truncation;
// fp32->bf16 SFPU RNE (ties-to-even); uint16->int32 SFPU zero-extend.
// int32->bf16 is composed as Typecast<Int32,Float32> -> Typecast<Float32,Float16_b> in ONE chain (shared
// dst-sync window): the direct Int32->Float16_b LLK arm rounds fp32->bf16 ties half-away (SFP_STOCH_RND),
// so it gets e.g. 257 -> 258 and 33685505 -> 33816576 wrong; torch rounds through float32 with RNE twice.
// With two SFPU elements the chain re-inits each per tile (their constant registers conflict).

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "g6_typecast_zones.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;
    constexpr auto in_df = dfb_l1_format<cb_input_tiles>();
    constexpr auto out_df = dfb_l1_format<cb_output_tiles>();
    constexpr InputSpec in_spec = input(cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block);
    constexpr OutputSpec out_spec = output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd);
    // Later blocks: the first block's call already emitted init + reconfig (Caller-owned from then on).
    constexpr InputSpec in_spec_hoisted = input(
        cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block, DataFormatReconfig::Disabled);
    constexpr OutputSpec out_spec_hoisted =
        output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled);

    DataflowBuffer dfb_in(cb_input_tiles);
    DataflowBuffer dfb_out(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    // Arm selection: int32 -> bf16 needs the two-step (two-rounding) composition, every other arm is one
    // Typecast element.
    constexpr uint32_t df_int32 = static_cast<uint32_t>(DataFormat::Int32);
    constexpr uint32_t df_fp32 = static_cast<uint32_t>(DataFormat::Float32);
    constexpr uint32_t df_bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
    constexpr bool two_step_int32_to_bf16 = in_df == df_int32 && out_df == df_bf16;
    auto convert = [&](auto owner_tag, const auto& shape, auto copy, auto pack) {
        constexpr InitReconfigOwner owner = decltype(owner_tag)::value;
        if constexpr (two_step_int32_to_bf16) {
            // Not boot-hoistable (two distinct SFPU inits): the chain owns init every block; later blocks'
            // specs still skip the (already-applied, unchanged) data-format reconfig.
            (void)owner;
            eltwise_chain<InitReconfigOwner::Chain>(
                shape, copy, Typecast<df_int32, df_fp32>{}, Typecast<df_fp32, df_bf16>{}, pack);
        } else {
            eltwise_chain<owner>(shape, copy, Typecast<in_df, out_df>{}, pack);
        }
    };
    using ChainOwned = std::integral_constant<InitReconfigOwner, InitReconfigOwner::Chain>;
    using CallerOwned = std::integral_constant<InitReconfigOwner, InitReconfigOwner::Caller>;

    G6_ZONE("CONVERT");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t block_extent = (core_tiles - done < block_tiles) ? (core_tiles - done) : block_tiles;
        const auto shape = IterationShape::tiles(block_extent).block_size(block_extent);
        if (done == 0) {
            convert(ChainOwned{}, shape, CopyTile<in_spec>{dfb_in}, PackTile<out_spec>{dfb_out});
        } else {
            convert(CallerOwned{}, shape, CopyTile<in_spec_hoisted>{dfb_in}, PackTile<out_spec_hoisted>{dfb_out});
        }
    }
}

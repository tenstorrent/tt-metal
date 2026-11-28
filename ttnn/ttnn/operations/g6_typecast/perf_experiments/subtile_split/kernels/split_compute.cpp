// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// subtile_split compute: the op's compute kernel (one chain call, Caller-owned inits, one CB handshake) on a
// TINY-TILE CB pair: cb_input_tiles / cb_output_tiles are declared with a 16x32 (NF = 2 faces) or 16x16 (NF = 1)
// tile, so hw_startup programs unpack, math datacopy and pack for NF faces only; the SFPU element runs the same
// passes as the op's Int32ToBf16Rne restricted to those NF faces (split_vector_mode<NF>: R = faces 0-1,
// None = face 0). NF = 4 is the op's kernel unchanged (32x32 tiles, RC).
// RAW-LLK note: TypecastNF / Int32ToBf16RneNF call _llk_math_eltwise_unary_sfpu_params_ directly because
// typecast_tile hard-codes VectorMode::RC (it would run the SFPU over 4 faces of a 2-face tile).

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "g6_typecast_zones.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "split_int32_bf16_sfpu.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t NF = get_compile_time_arg_val(3);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;
    constexpr auto in_df = dfb_l1_format<cb_input_tiles>();
    constexpr auto out_df = dfb_l1_format<cb_output_tiles>();
    constexpr InputSpec in_spec_hoisted = input(
        cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block, DataFormatReconfig::Disabled);
    constexpr OutputSpec out_spec_hoisted =
        output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled);

    DataflowBuffer dfb_in(cb_input_tiles);
    DataflowBuffer dfb_out(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    constexpr uint32_t df_int32 = static_cast<uint32_t>(DataFormat::Int32);
    constexpr uint32_t df_bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
    constexpr bool int32_to_bf16 = in_df == df_int32 && out_df == df_bf16;
    auto element = []() {
        if constexpr (int32_to_bf16) {
            return Int32ToBf16RneNF<NF>{};
        } else {
            return TypecastNF<in_df, out_df, NF>{};
        }
    };
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

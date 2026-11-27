// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast compute: convert_block.
//
// One eltwise_chain (CopyTile -> Typecast -> PackTile) per block of `block_tiles` pages: the input
// block is waited and the output block reserved once (Upfront), popped / pushed once (AtEnd), and
// the chain walks the block in DEST groups of DEST_AUTO_LIMIT tiles. The LLK formats come from the
// CB formats, so one kernel serves every (input, output) pair the host configures.
//
// The chain's one-time setup is caller-owned (InitReconfigOwner::Caller, the hoist the chain header
// documents): copy_init + typecast_tile_init are issued once per kernel rather than once per block,
// because the CB formats never change. compute_kernel_hw_startup(in, out) already programs the
// unpack and pack formats, so every spec disables reconfig.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t in_df = ckl::dfb_l1_format<cb_input_tiles>();
    constexpr uint32_t out_df = ckl::dfb_l1_format<cb_output_tiles>();
    constexpr uint32_t dest_group = ckl::DEST_AUTO_LIMIT;

    using ConvertInput = ckl::CopyTile<ckl::input(
        cb_input_tiles,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Disabled)>;
    using ConvertOutput = ckl::PackTile<ckl::output(
        cb_output_tiles, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>;
    using Convert = ckl::Typecast<in_df, out_df>;

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);
    copy_init(cb_input_tiles);
    typecast_tile_init<in_df, out_df>();

    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t block_extent = (core_tiles - done) < block_tiles ? (core_tiles - done) : block_tiles;
#if defined(PROFILE_KERNEL)
        {
            // Separates starvation from conversion; the chain's own Upfront wait then returns at once.
            DeviceZoneScopedN("TC-CV-WAIT");
            input_tiles.wait_front(block_extent);
        }
#endif
        {
            DeviceZoneScopedN("TC-CV-BLOCK");
            ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                ckl::IterationShape::tiles(block_extent).block_size(dest_group),
                ConvertInput{input_tiles},
                Convert{},
                ConvertOutput{output_tiles});
        }
    }
}

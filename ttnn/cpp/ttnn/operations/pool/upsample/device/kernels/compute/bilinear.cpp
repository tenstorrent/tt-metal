// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

template <uint32_t tiles_per_reduction>
inline void reduce_h_fused(DataflowBuffer in_dfb, DataflowBuffer scalar_dfb, DataflowBuffer out_dfb) {
    const uint32_t in_cb_id = in_dfb.get_id();
    const uint32_t in_scalar_cb_id = scalar_dfb.get_id();
    const uint32_t out_cb_id = out_dfb.get_id();
    tile_regs_acquire();
    in_dfb.wait_front(4);

    // Template parameters for unpack_tilizeA_B_block:
    constexpr bool use_neginf_srcA = false;  // Don't use negative infinity for source A
    constexpr bool reload_srcB = true;       // Reload source B (bilinear weights) for each operation
    constexpr bool zero_srcA = false;        // Don't zero source A
    constexpr bool zero_srcA_reduce = true;  // Zero source A for reduce operation

    // Function parameters:
    constexpr uint32_t scalar_tile_idx = 0;  // Tile index for scalar CB (only 1 tile of weights loaded)
    constexpr uint32_t num_faces = 2;  // Unpack 2 faces (top faces contain 4 rows needed for bilinear interpolation)

    unpack_tilizeA_B_block<use_neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        in_cb_id, in_scalar_cb_id, tiles_per_reduction, scalar_tile_idx);
    for (uint32_t c_i = 0; c_i < tiles_per_reduction; ++c_i) {
        reduce_tile_math<REDUCE_OP, REDUCE_DIM>(
            c_i, num_faces);  // Reduce the 2 faces (containing 4 rows for bilinear interpolation)
    }
    in_dfb.pop_front(4);

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dest<tiles_per_reduction>(out_cb_id); /* face geometry comes from out_cb metadata */
    tile_regs_release();

    // [Quasar bring-up: bilinear upsample Metal-2.0 port]
    // was a hand-rolled advance of the legacy `cb_interface` write pointer
    // (llk_push_pages_bilinear). That global does not exist on Gen2: the trisc2 link fails
    // with "undefined reference to `cb_interface`". The DFB entry size IS out_cb_pagesize
    // (TILE_WIDTH * element_size), so pushing tiles_per_reduction entries advances the
    // write pointer by exactly the same number of bytes.
    PACK(out_dfb.push_back(tiles_per_reduction));
}

void kernel_main() {
    // Runtime argument - work count for this core
    uint32_t nsticks_per_core_by_nblocks = get_arg(args::nsticks_per_core_by_nblocks);

    constexpr uint32_t in_ntiles_c = get_arg(args::in_ntiles_c);
    constexpr uint32_t out_ntiles_c = get_arg(args::out_ntiles_c);
    constexpr uint32_t blocks = get_arg(args::blocks);
    constexpr uint32_t input_block_size_bytes = get_arg(args::input_block_size_bytes);

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    constexpr uint32_t num_output_tiles = out_ntiles_c;  //* nblocks;

    constexpr bool use_neginf_srcA = false;  // Don't use negative infinity for source A
    constexpr bool zero_srcA_reduce = true;  // Zero source A for reduce operation

    DataflowBuffer tilize_reduce_dfb0(dfb::tilize_reduce_cb_0);
    DataflowBuffer tilize_reduce_dfb1(dfb::tilize_reduce_cb_1);
    DataflowBuffer scalar_dfb_1(dfb::in_scalar_cb_1);
    DataflowBuffer scalar_dfb_2(dfb::in_scalar_cb_2);
    DataflowBuffer out_dfb(dfb::out_cb);

    compute_kernel_hw_startup(dfb::tilize_reduce_cb_0, dfb::in_scalar_cb_1, dfb::out_cb);
    tilizeA_B_reduce_init<use_neginf_srcA, zero_srcA_reduce>(
        dfb::tilize_reduce_cb_0, dfb::in_scalar_cb_1, max_tiles_per_iter);
    pack_untilize_dest_init<max_tiles_per_iter>(dfb::out_cb); /* face geometry comes from out_cb metadata */
    for (uint32_t i = 0; i < nsticks_per_core_by_nblocks; i++) {
        DataflowBuffer cur_in_dfb = (i % 2 == 0) ? tilize_reduce_dfb0 : tilize_reduce_dfb1;
        DataflowBuffer cur_scalar_dfb = (i % 2 == 0) ? scalar_dfb_1 : scalar_dfb_2;

        for (uint32_t j = 0; j < blocks - 1; j++) {
            reduce_h_fused<max_tiles_per_iter>(cur_in_dfb, cur_scalar_dfb, out_dfb);
            cur_scalar_dfb.pop_front(1);
        }
        reduce_h_fused<partial_iter_output_tiles>(cur_in_dfb, cur_scalar_dfb, out_dfb);
        cur_scalar_dfb.pop_front(1);
    }
}  // void kernel_main()

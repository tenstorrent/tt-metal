// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/compute/pack_untilize.h"
#include "internal/circular_buffer_interface.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Push 1 stick or partial stick to a cb (a (partial) stick consists of num_pages pages, in our case, size of a page is
// the width of a tile (setup in program factory))
inline void llk_push_pages_bilinear(const std::int32_t operand, const std::int32_t num_pages) {
    std::uint32_t output = operand;
    std::uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(output).fifo_wr_ptr += num_words;
    get_local_cb_interface(output).fifo_wr_tile_ptr = 0;

    if (get_local_cb_interface(output).fifo_wr_ptr >= get_local_cb_interface(output).fifo_limit) {
        get_local_cb_interface(output).fifo_wr_ptr -= get_local_cb_interface(output).fifo_size;
    }
}

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
        reduce_tile_math<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
            c_i, num_faces);  // Reduce the 2 faces (containing 4 rows for bilinear interpolation)
    }
    in_dfb.pop_front(4);

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dest<tiles_per_reduction>(out_cb_id); /* face geometry comes from out_cb metadata */
    tile_regs_release();

    PACK(llk_push_pages_bilinear(out_cb_id, tiles_per_reduction));
}

template <uint32_t in_ntiles_c, uint32_t blocks>
TT_KERNEL void bilinear(uint32_t nsticks_per_core_by_nblocks) {
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    constexpr bool use_neginf_srcA = false;  // Don't use negative infinity for source A
    constexpr bool zero_srcA_reduce = true;  // Zero source A for reduce operation

    DataflowBuffer tilize_reduce_dfb0(dfb::tilize_reduce0);
    DataflowBuffer tilize_reduce_dfb1(dfb::tilize_reduce1);
    DataflowBuffer scalar_dfb_1(dfb::scalar0);
    DataflowBuffer scalar_dfb_2(dfb::scalar1);
    DataflowBuffer out_dfb(dfb::output);

    compute_kernel_hw_startup(dfb::tilize_reduce0, dfb::scalar0, dfb::output);
    tilizeA_B_reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL, use_neginf_srcA, zero_srcA_reduce>(
        dfb::tilize_reduce0, dfb::scalar0, max_tiles_per_iter);
    pack_untilize_dest_init<max_tiles_per_iter>(dfb::output); /* face geometry comes from out_cb metadata */
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
}

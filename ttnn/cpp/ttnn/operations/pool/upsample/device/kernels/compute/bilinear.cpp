// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/compute/pack_untilize.h"
#include "internal/circular_buffer_interface.h"
#include "api/dataflow/dataflow_buffer.h"
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
        reduce_tile_math<REDUCE_OP, REDUCE_DIM>(
            c_i, num_faces);  // Reduce the 2 faces (containing 4 rows for bilinear interpolation)
    }
    in_dfb.pop_front(4);

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dest<tiles_per_reduction>(out_cb_id); /* face geometry comes from out_cb metadata */
    tile_regs_release();

    PACK(llk_push_pages_bilinear(out_cb_id, tiles_per_reduction));
}

void kernel_main() {
    // Runtime argument - work count for this core
    uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(0);

    constexpr uint32_t tilize_reduce_cb_0 = get_compile_time_arg_val(0);
    constexpr uint32_t tilize_reduce_cb_1 = get_compile_time_arg_val(1);
    constexpr uint32_t in_scalar_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t in_scalar_cb_id2 = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(5);
    constexpr uint32_t in_ntiles_hwc = get_compile_time_arg_val(6);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(7);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(8);
    constexpr uint32_t blocks = get_compile_time_arg_val(9);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(10);

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    constexpr uint32_t num_output_tiles = out_ntiles_c;  //* nblocks;

    constexpr bool use_neginf_srcA = false;  // Don't use negative infinity for source A
    constexpr bool zero_srcA_reduce = true;  // Zero source A for reduce operation

    DataflowBuffer tilize_reduce_dfb0(tilize_reduce_cb_0);
    DataflowBuffer tilize_reduce_dfb1(tilize_reduce_cb_1);
    DataflowBuffer scalar_dfb_1(in_scalar_cb_id1);
    DataflowBuffer scalar_dfb_2(in_scalar_cb_id2);
    DataflowBuffer out_dfb(out_cb_id);

    tilizeA_B_reduce_init<use_neginf_srcA, zero_srcA_reduce>(
        tilize_reduce_cb_0, in_scalar_cb_id1, max_tiles_per_iter, out_cb_id);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id); /* face geometry comes from out_cb metadata */
    uint32_t i = 0;

    // Two output pixels per DEST round trip when they fit: a single channel block (the whole stick in one reduction)
    // and room for both pixels' tiles in DEST (bf16 DEST, 8 tiles). Pixel i comes from the reader's CB and weights,
    // pixel i + 1 from the writer's (the two dataflow RISCs alternate pixels). The second pixel's reduce writes DEST
    // tiles [max_tiles_per_iter, 2 * max_tiles_per_iter) and is packed from there (tile_dst_rt_offset is a DEST tile
    // index). CB ids stay compile-time constants: a generic k-loop with runtime CB selection measured slower (78.7 vs
    // 63.5 us on EVO50 96x416x32), and so did larger groups (4 / 8 pixels: the input CBs' buffering stalls the
    // readers).
    constexpr bool pair_pixels = (blocks == 1) && (2 * max_tiles_per_iter <= 8);
    if constexpr (pair_pixels) {
        constexpr bool reload_srcB = true;
        constexpr bool zero_srcA = false;
        constexpr uint32_t num_faces = 2;
        for (; i + 1 < nsticks_per_core_by_nblocks; i += 2) {
            tile_regs_acquire();
            tilize_reduce_dfb0.wait_front(4);
            unpack_tilizeA_B_block<use_neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
                tilize_reduce_cb_0, in_scalar_cb_id1, max_tiles_per_iter, 0);
            for (uint32_t c_i = 0; c_i < max_tiles_per_iter; ++c_i) {
                reduce_tile_math<REDUCE_OP, REDUCE_DIM>(c_i, num_faces);
            }
            tilize_reduce_dfb0.pop_front(4);
            scalar_dfb_1.pop_front(1);
            tilize_reduce_dfb1.wait_front(4);
            unpack_tilizeA_B_block<use_neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
                tilize_reduce_cb_1, in_scalar_cb_id2, max_tiles_per_iter, 0);
            for (uint32_t c_i = 0; c_i < max_tiles_per_iter; ++c_i) {
                reduce_tile_math<REDUCE_OP, REDUCE_DIM>(max_tiles_per_iter + c_i, num_faces);
            }
            tilize_reduce_dfb1.pop_front(4);
            scalar_dfb_2.pop_front(1);
            tile_regs_wait();
            tile_regs_commit();
            pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, 0);
            PACK(llk_push_pages_bilinear(out_cb_id, max_tiles_per_iter));
            pack_untilize_dest<max_tiles_per_iter>(
                out_cb_id, 1, 0, max_tiles_per_iter);  // DEST tile offset of pixel i+1
            PACK(llk_push_pages_bilinear(out_cb_id, max_tiles_per_iter));
            tile_regs_release();
        }
    }

    for (; i < nsticks_per_core_by_nblocks; i++) {
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

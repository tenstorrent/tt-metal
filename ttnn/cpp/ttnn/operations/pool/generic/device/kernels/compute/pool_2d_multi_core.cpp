// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/common.h"
#include "tools/profiler/kernel_profiler.hpp"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

template <uint32_t num_tiles, uint32_t split_reader, uint32_t num_pages_to_8>
inline void eltwise_mul_tiles(
    const uint32_t in_cb_id_0,
    const uint32_t in_cb_id_1,
    const uint32_t weight_cb_id,
    const uint32_t in_stick_index,
    const uint32_t mul_cb_id) {
    DeviceZoneScopedN("eltwise_mul_tiles init");
    const uint32_t curr_in_cb_id = (split_reader && (in_stick_index & 0x1)) ? in_cb_id_1 : in_cb_id_0;
    cb_reserve_back(mul_cb_id, num_pages_to_8);  // packer zabo
    {
        DeviceZoneScopedN("eltwise_mul_tiles init");
        UNPACK((llk_unpack_tilize_uninit(curr_in_cb_id)));
        PACK((pack_untilize_uninit(mul_cb_id)));
        mul_tiles_init(curr_in_cb_id, weight_cb_id);
    }
    for (uint32_t j = 0; j < num_pages_to_8; j++) {
        for (uint32_t i = 0; i < num_tiles; ++i) {
            cb_wait_front(curr_in_cb_id, 1);  // unpacker zabo

            // UNPACK(( DPRINT << "now it is tile number: " << j << ENDL() ));
            // UNPACK(( tt::compute::common::print_full_tile(weight_cb_id, 0) ));

            tile_regs_acquire();

            // UNPACK(( tt::compute::common::print_full_tile(weight_cb_id, i) ));
            mul_tiles(curr_in_cb_id, weight_cb_id, 0, 0, j * num_tiles + i);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(j * num_tiles + i, mul_cb_id, j * num_tiles + i);
            tile_regs_release();
            cb_pop_front(curr_in_cb_id, 1);
            cb_push_back(mul_cb_id, 1);
        }
    }
}

template <
    uint32_t num_output_tiles,
    bool is_partial_tile,
    uint32_t split_reader,
    uint32_t unpA_face_r_dim,
    uint32_t num_faces_in_tile,
    bool neginf_srca_maxpool,
    bool zero_srca_avgpool,
    uint32_t num_pages_to_8>
inline void reduce_h_fused(
    const uint32_t in_cb_id_0,
    const uint32_t in_cb_id_1,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id,
    const uint32_t mul_cb_id) {
    DeviceZoneScopedN("reduce_h_fused");
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t num_output_faces = (is_partial_tile ? 1 : 2);

    const uint32_t curr_in_cb_id = mul_cb_id;

    // for (uint32_t i = 0; i < num_pages_to_8; i++) {
    //     UNPACK(( DPRINT << "now it is tile number: " << i << ENDL() ));
    //     UNPACK(( tt::compute::common::print_full_tile(curr_in_cb_id, i) ));
    // }

    // tilize krece od pocetka cb-a i ne moze da se offsetuje
    // obican unpack moze
    // obican tilize moze
    // u kombinaciji ne moze
    {
        DeviceZoneScopedN("reduce init");
        UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
            mul_cb_id, in_scalar_cb_id, num_output_tiles, num_faces_in_tile, unpA_face_r_dim, 1)));
        MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>()));
        PACK((llk_pack_untilize_init<num_output_tiles, num_output_tiles, false, false, TILE_C_DIM>(
            out_cb_id, num_out_rows, num_faces_in_tile)));
        PACK((llk_init_packer_dest_offset_registers<true, false>()));
    }

    for (uint32_t j = 0; j < num_pages_to_8; j++) {
        DeviceZoneScopedN("iteration");
        {
            DeviceZoneScopedN("wait");
            cb_reserve_back(out_cb_id, 1);
            cb_wait_front(curr_in_cb_id, 1);
        }
        tile_regs_acquire();
        {
            DeviceZoneScopedN("unpack reduce");
            unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                curr_in_cb_id, in_scalar_cb_id, num_output_tiles, 0, num_faces_in_tile, unpA_face_r_dim);
            for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
                reduce_tile_math(c_i, num_faces_in_tile);
            }
        }

        tile_regs_wait();
        tile_regs_commit();

        cb_pop_front(curr_in_cb_id, 1);
        // num_output_tiles takodje govori koja je sirina tensora
        pack_untilize_dest<num_output_tiles>(
            out_cb_id, 1 /*out_subblock_h*/, 0, num_out_rows, num_output_faces); /* pack 1 row (1x16 or 1x32) */
        tile_regs_release();
        cb_push_back(out_cb_id, 1);
    }

    // dprint_tensix_dest_reg(1);

    // pack -> zero out dest

    // PACK(( DPRINT << "*************" << ENDL() ));
    // PACK(( tt::compute::common::print_full_tile(out_cb_id, 0, true) ));
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t nsticks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(10);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(11);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t weight_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t mul_cb_id = get_compile_time_arg_val(14);

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = window_size_hw > 16 ? 4 : (is_partial_tile ? 1 : 2);
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    constexpr uint32_t num_pages_to_8 = 8 / in_ntiles_c;

    // In case we have <=16 sticks we will use only upper two faces of the tile.
    // In this case we can configure reduce to only process as many rows as needed.
    // In case #sticks > 16 we need bottom two faces as well, and we need to configure reduce to
    // process all rows per face. In the case we rely on reader kernel to put "clear value"
    // in datums which are not used.
    constexpr uint32_t face_r_dim = window_size_hw > 16 ? 16 : window_size_hw;

    // mul_tiles_init(in_cb_id_0, weight_cb_id);
    // if (split_reader) {
    //     mul_tiles_init(in_cb_id_1, weight_cb_id);
    // }

    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        mul_cb_id, in_scalar_cb_id_0, max_tiles_per_iter, out_cb_id, num_faces_in_tile, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_tile);

    // tilize reconfiguration is needed if we have more than one block and the number of tiles
    // is not a multiple of MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig_needed = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0;
    if (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }
    for (uint32_t i = 0; i < nsticks_per_core / 8; ++i) {
        const uint32_t curr_scalar_cb_id =
            (split_reader && (i & 0x1) && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
        }
        // perform the reduction over the first N - 1 whole chunks
        if constexpr (tilize_reconfig_needed) {
            UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                mul_cb_id, curr_scalar_cb_id, max_tiles_per_iter, num_faces_in_tile, face_r_dim, 1)));
        }

        for (uint32_t b_i = 0; b_i < in_nblocks_c - 1; ++b_i) {
            eltwise_mul_tiles<max_tiles_per_iter, split_reader, num_pages_to_8>(
                in_cb_id_0, in_cb_id_1, weight_cb_id, i, mul_cb_id);

            reduce_h_fused<
                max_tiles_per_iter,
                is_partial_tile,
                split_reader,
                face_r_dim,
                num_faces_in_tile,
                neginf_srca_maxpool,
                zero_srca_avgpool,
                num_pages_to_8>(in_cb_id_0, in_cb_id_1, curr_scalar_cb_id, i, out_cb_id, mul_cb_id);
        }

        if constexpr (tilize_reconfig_needed) {
            UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                mul_cb_id, curr_scalar_cb_id, partial_iter_output_tiles, num_faces_in_tile, face_r_dim, 1)));
        }
        // perform the reduction over the either whole or partial chunk N

        // DPRINT << "partial_iter_output_tiles: " << partial_iter_output_tiles << ENDL();

        eltwise_mul_tiles<partial_iter_output_tiles, split_reader, num_pages_to_8>(
            in_cb_id_0, in_cb_id_1, weight_cb_id, i, mul_cb_id);

        reduce_h_fused<
            partial_iter_output_tiles,
            is_partial_tile,
            split_reader,
            face_r_dim,
            num_faces_in_tile,
            neginf_srca_maxpool,
            zero_srca_avgpool,
            num_pages_to_8>(in_cb_id_0, in_cb_id_1, curr_scalar_cb_id, i, out_cb_id, mul_cb_id);
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
    if constexpr (one_scalar_per_core) {
        cb_pop_front(in_scalar_cb_id_0, 1);
    }
}

}  // namespace NAMESPACE

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "reader_pool2d_sharded_common.hpp"
#define ENABLE_DEBUG_PRINT 1

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t num_top_left_indexes = get_compile_time_arg_val(2);
    constexpr uint32_t in_c = get_compile_time_arg_val(3);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(5);
    constexpr uint32_t out_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_w = get_compile_time_arg_val(7);
    constexpr uint32_t is_out_tiled = get_compile_time_arg_val(8);
    constexpr uint32_t BYTES_PER_ELEM = 2;  // bf16
    // for (int i = 0; i<32;i++){
    //     DPRINT<<"["<<i<<"]"<<ENDL();
    //     tt::data_movement::common ::print_bf16_pages(get_write_ptr(cb_src)+64*i*BYTES_PER_ELEM, 64, BYTES_PER_ELEM);
    // }
    DPRINT << "cb_src:" << cb_src << ENDL();
    DPRINT << "cb_dst:" << cb_dst << ENDL();
    DPRINT << "num_top_left_indexes:" << num_top_left_indexes << ENDL();
    DPRINT << "in_c:" << in_c << ENDL();
    DPRINT << "in_nblocks_c:" << in_nblocks_c << ENDL();
    DPRINT << "in_ntiles_c:" << in_ntiles_c << ENDL();
    DPRINT << "out_h:" << out_h << ENDL();
    DPRINT << "out_w:" << out_w << ENDL();
    DPRINT << "is_out_tiled:" << is_out_tiled << ENDL();

    constexpr uint32_t face_dim = 16;
    constexpr uint32_t face_size = face_dim * face_dim;  // 256
    constexpr uint32_t tile_face_dim = 2;
    constexpr uint32_t tile_size = face_size * tile_face_dim * tile_face_dim;  // 1024

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;
    // DPRINT << "partial_iter_output_tiles:" << partial_iter_output_tiles << ENDL();

    uint32_t out_l1_write_addr_base = get_write_ptr(cb_dst);

    for (uint32_t i = 0; i < num_top_left_indexes; ++i) {  // read row by row

        for (uint32_t c_i = 0; c_i < in_nblocks_c - 1; ++c_i) {
            uint32_t in_l1_read_addr = get_read_ptr(cb_src);
            cb_wait_front(cb_src, max_tiles_per_iter);
            cb_reserve_back(cb_dst, max_tiles_per_iter);
            if constexpr (is_out_tiled) {
                // elements within a face are row major
                // faces within a tile are row major
                // tiles within a tensor are row major
                uint32_t face_nhw = i / face_dim;
                uint32_t intra_face_nhw = i % face_dim;
                constexpr uint32_t max_faces_per_row = max_tiles_per_iter * tile_face_dim;
                // DPRINT  << "max_faces_per_row:" << max_faces_per_row << ENDL();
                for (uint32_t face_c = 0; face_c < max_faces_per_row; ++face_c) {
                    uint32_t tile_nhw = face_nhw / tile_face_dim;
                    uint32_t tile_c = c_i * max_tiles_per_iter + face_c / tile_face_dim;
                    uint32_t face_mod_nhw = face_nhw % tile_face_dim;
                    uint32_t face_mod_c = face_c % tile_face_dim;
                    uint32_t tile_offset = (tile_nhw * in_ntiles_c * tile_size + tile_c * tile_size) * BYTES_PER_ELEM;
                    uint32_t face_offset =
                        (face_mod_nhw * face_size * tile_face_dim + face_mod_c * face_size) * BYTES_PER_ELEM;
                    uint32_t intra_face_offset = intra_face_nhw * face_dim * BYTES_PER_ELEM;
                    noc_async_read_one_packet(
                        get_noc_addr(in_l1_read_addr) + face_c * face_dim * BYTES_PER_ELEM,
                        out_l1_write_addr_base + tile_offset + face_offset + intra_face_offset,
                        face_dim * BYTES_PER_ELEM);
                }
            } else {
                uint32_t out_l1_write_addr = get_write_ptr(cb_dst);
                noc_async_read_one_packet(
                    get_noc_addr(in_l1_read_addr),
                    out_l1_write_addr,
                    max_tiles_per_iter * (in_c == 16 ? 16 : 32) * BYTES_PER_ELEM);
            }
            noc_async_read_barrier();
            cb_push_back(cb_dst, max_tiles_per_iter);
            cb_pop_front(cb_src, max_tiles_per_iter);
        }

        // Rest partial tiles
        uint32_t in_l1_read_addr = get_read_ptr(cb_src);
        // DPRINT << "COPY INPUT:" << ENDL();
        // tt::data_movement::common ::print_bf16_pages(in_l1_read_addr, 32 * partial_iter_output_tiles, 1);

        cb_wait_front(cb_src, partial_iter_output_tiles);
        cb_reserve_back(cb_dst, partial_iter_output_tiles);

        if constexpr (is_out_tiled) {
            // elements within a face are row major
            // faces within a tile are row major
            // tiles within a tensor are row major
            uint32_t face_nhw = i / face_dim;
            uint32_t intra_face_nhw = i % face_dim;
            constexpr uint32_t partial_faces_per_row = partial_iter_output_tiles * tile_face_dim;
            // DPRINT  << "partial_faces_per_row:" << partial_faces_per_row << ENDL();
            for (uint32_t face_c = 0; face_c < partial_faces_per_row; ++face_c) {
                uint32_t tile_nhw = face_nhw / tile_face_dim;
                uint32_t tile_c = (in_nblocks_c - 1) * max_tiles_per_iter + face_c / tile_face_dim;
                uint32_t face_mod_nhw = face_nhw % tile_face_dim;
                uint32_t face_mod_c = face_c % tile_face_dim;
                uint32_t tile_offset = (tile_nhw * in_ntiles_c * tile_size + tile_c * tile_size) * BYTES_PER_ELEM;
                uint32_t face_offset =
                    (face_mod_nhw * face_size * tile_face_dim + face_mod_c * face_size) * BYTES_PER_ELEM;
                uint32_t intra_face_offset = intra_face_nhw * face_dim * BYTES_PER_ELEM;
                noc_async_read_one_packet(
                    get_noc_addr(in_l1_read_addr) + face_c * face_dim * BYTES_PER_ELEM,
                    out_l1_write_addr_base + tile_offset + face_offset + intra_face_offset,
                    face_dim * BYTES_PER_ELEM);
                // DPRINT  <<"["<< i<< "]face_c:" << face_c << ENDL();
                // tt::data_movement::common ::print_bf16_pages(out_l1_write_addr_base+ tile_offset + face_offset +
                // intra_face_offset, face_dim, BYTES_PER_ELEM);
            }
        } else {
            uint32_t out_l1_write_addr = get_write_ptr(cb_dst);
            noc_async_read_one_packet(
                get_noc_addr(in_l1_read_addr),
                out_l1_write_addr,
                partial_iter_output_tiles * (in_c == 16 ? 16 : 32) *
                    BYTES_PER_ELEM);  // TODO: note: change. take care partial block/tiles.
        }
        noc_async_read_barrier();

        cb_push_back(cb_dst, partial_iter_output_tiles);
        cb_pop_front(cb_src, partial_iter_output_tiles);

        // DPRINT << "i: " << i << ENDL();
        // tt::data_movement::common ::print_bf16_pages(out_l1_write_addr_base, 32, 32);
    }

    // tt::data_movement::common ::print_bf16_pages(out_l1_write_addr_base, in_c, 32);

}  // kernel_main()

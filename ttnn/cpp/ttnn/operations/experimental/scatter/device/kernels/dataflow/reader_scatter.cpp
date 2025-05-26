// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "dprint.h"

#include "../scatter_common.hpp"

namespace {

template <bool is_dram>
FORCE_INLINE void read_wt_tiles(
    IAGF<is_dram> addr_gtor, const uint32_t& cb, const uint32_t& wt_tiles, const uint32_t& ht_offset = 0) {
    for (uint32_t tile = 0; tile < wt_tiles; ++tile) {
        cb_reserve_back(cb, ONE_TILE);
        const uint32_t l1_addr = get_write_ptr(cb);
        noc_async_read_tile(ht_offset * wt_tiles + tile, addr_gtor, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb, ONE_TILE);
    }
}

// O(n^2), but no constraints
template <
    typename number_type,
    typename index_type,
    bool input_tensor_is_dram,
    bool index_tensor_is_dram,
    bool source_tensor_is_dram,
    bool output_tensor_is_dram>
FORCE_INLINE void scatter_along_whole_axis(
    const uint32_t& ht_offset,
    const uint32_t& Wt_input,
    const uint32_t& logical_width,
    const uint32_t& logical_height,
    const uint32_t& Wt_index,
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb,
    const IAGF<input_tensor_is_dram>& input_addr_gtor,
    const IAGF<index_tensor_is_dram>& index_addr_gtor,
    const IAGF<source_tensor_is_dram>& source_addr_gtor,
    const IAGF<output_tensor_is_dram>& output_addr_gtor) {
    for (uint32_t tile_input = 0; tile_input < Wt_input; ++tile_input) {
        // cb_wait_front(input_cb, ONE_TILE);
        cb_reserve_back(output_cb, ONE_TILE);
        // read an input tile + get fresh pointers
        read_wt_tiles<input_tensor_is_dram>(input_addr_gtor, input_cb, ONE_TILE, ht_offset);
        const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
        const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
        volatile tt_l1_ptr number_type* input_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
        volatile tt_l1_ptr number_type* output_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);

        // copy WIP input tile into output
        for (uint32_t tile_input_inner = 0; tile_input_inner < tt::constants::TILE_HW; ++tile_input_inner) {
            output_l1_ptr[tile_input_inner] = input_l1_ptr[tile_input_inner];
        }
        // scatter Wt_index tiles onto ONE_TILE from Wt_input
        for (uint32_t tile_index_w = 0; tile_index_w < Wt_index; ++tile_index_w) {
            // read index and source tiles + get fresh pointers
            read_wt_tiles<index_tensor_is_dram>(index_addr_gtor, index_cb, ONE_TILE, ht_offset);
            read_wt_tiles<source_tensor_is_dram>(source_addr_gtor, source_cb, ONE_TILE, ht_offset);
            const uint32_t index_l1_read_addr = get_read_ptr(index_cb);
            const uint32_t source_l1_read_addr = get_read_ptr(source_cb);
            volatile tt_l1_ptr index_type* index_l1_ptr =
                reinterpret_cast<volatile tt_l1_ptr index_type*>(index_l1_read_addr);
            volatile tt_l1_ptr number_type* source_l1_ptr =
                reinterpret_cast<volatile tt_l1_ptr number_type*>(source_l1_read_addr);
            for (uint32_t face_x = 0; face_x < 2; ++face_x) {
                for (uint32_t face_y = 0; face_y < 2; ++face_y) {
                    for (uint32_t scalar_x = 0; scalar_x < 16; ++scalar_x) {
                        const uint32_t width_scalar_index = get_width_scalar_index(tile_index_w, face_x, scalar_x);
                        // break sooner if the pointer went past logical width
                        if (width_scalar_index >= logical_width) {
                            continue;
                        }
                        for (uint32_t scalar_y = 0; scalar_y < 16; ++scalar_y) {
                            // get global coords + assert
                            const uint32_t height_scalar_index = get_height_scalar_index(ht_offset, face_y, scalar_y);
                            // everything afterward is padded values (past logical height)
                            if (height_scalar_index >= logical_height) {
                                continue;
                            }

                            // get scatter info
                            volatile index_type& index_value =
                                tile_guts<index_type>(index_l1_ptr, face_x, face_y, scalar_x, scalar_y, tile_index_w);
                            // check if index value targets currently chosen input tile (tile_input)
                            // shall `index_type` be used for those variables?
                            const uint32_t dest_tile_id_in_row = index_value >> 5;
                            if (dest_tile_id_in_row != tile_input) {
                                continue;
                            }
                            volatile number_type& source_value =
                                tile_guts<number_type>(source_l1_ptr, face_x, face_y, scalar_x, scalar_y, tile_index_w);
                            const uint32_t x_index_in_tile = index_value & 31;
                            const uint32_t dest_scalar_x =
                                (x_index_in_tile < 16) ? x_index_in_tile : (x_index_in_tile - 16);
                            const uint32_t dest_face_id_x = (x_index_in_tile < 16) ? 0 : 1;

                            // scatter the value
                            tile_guts<number_type>(
                                output_l1_ptr, dest_face_id_x, face_y, dest_scalar_x, scalar_y, dest_tile_id_in_row) =
                                source_value;
                        }
                    }
                }
            }
        }

        // push to the output
        cb_push_back(output_cb, ONE_TILE);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const DataFormat input_data_format{get_dataformat(ctas.input_tensor_cb)};

    const auto input_addr_gtor{make_addr_gtor<ctas.input_tensor_is_dram>(ctas.input_tensor_cb, ctas.input_tensor_addr)};
    const auto index_addr_gtor{make_addr_gtor<ctas.index_tensor_is_dram>(ctas.index_tensor_cb, ctas.index_tensor_addr)};
    const auto source_addr_gtor{
        make_addr_gtor<ctas.source_tensor_is_dram>(ctas.source_tensor_cb, ctas.source_tensor_addr)};
    const auto output_addr_gtor{
        make_addr_gtor<ctas.output_tensor_is_dram>(ctas.output_tensor_cb, ctas.output_tensor_addr)};

    const uint32_t start_ht_id = get_arg_val<uint32_t>(0);
    const uint32_t ht_per_core = get_arg_val<uint32_t>(1);

    using input_std_type = std_type_t<get_dataformat(ctas.input_tensor_cb)>;
    using index_std_type = std_type_t<get_dataformat(ctas.index_tensor_cb)>;

    for (uint32_t h = start_ht_id; h < start_ht_id + ht_per_core; ++h) {
        scatter_along_whole_axis<
            input_std_type,
            index_std_type,
            ctas.input_tensor_is_dram,
            ctas.index_tensor_is_dram,
            ctas.source_tensor_is_dram,
            ctas.output_tensor_is_dram>(
            h,
            ctas.Wt_input,
            ctas.logical_index_width,
            ctas.logical_index_height,
            ctas.Wt_index,
            ctas.input_tensor_cb,
            ctas.index_tensor_cb,
            ctas.source_tensor_cb,
            ctas.output_tensor_cb,
            input_addr_gtor,
            index_addr_gtor,
            source_addr_gtor,
            output_addr_gtor);
    }
}

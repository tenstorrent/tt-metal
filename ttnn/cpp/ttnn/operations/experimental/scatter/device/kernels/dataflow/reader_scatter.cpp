// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "dprint.h"

#include "../scatter_common.hpp"

namespace {

template <bool is_dram>
FORCE_INLINE void read_nth_tile_in_Wt_tiles(
    IAGF<is_dram> addr_gtor,
    const uint32_t& cb,
    const uint32_t& Wt_tiles,
    const uint32_t& ht_offset,
    const uint32_t& tile_num_in_row) {
    cb_reserve_back(cb, ONE_TILE);
    const uint32_t l1_addr = get_write_ptr(cb);
    noc_async_read_tile(ht_offset * Wt_tiles + tile_num_in_row, addr_gtor, l1_addr);
    noc_async_read_barrier();
    cb_push_back(cb, ONE_TILE);
}

// several comments on the algorithm
// computational complexity: O(input_shape[scatter_axis] * index_shape[scatter_axis])
// memory complexity: O(3 * tile_size(input_cb) + tile_size(index_cb)) -> O(1)
// this algorithm courses through each tile along the whole scatter axis (index_shape[scatter_axis]) in the index tensor
// (and source tensor on parallel) for each input tile along the whole scatter axis (input_shape[scatter_axis])
template <
    typename number_type,
    typename index_type,
    bool input_tensor_is_dram,
    bool index_tensor_is_dram,
    bool source_tensor_is_dram>
FORCE_INLINE void scatter_along_whole_axis(
    const uint32_t& ht_offset,
    const uint32_t& Wt_input,
    const uint32_t& logical_index_width,
    const uint32_t& logical_index_height,
    const uint32_t& Wt_index,
    const uint32_t& counter,
    const uint32_t& pad_scalar_offset,
    const uint32_t& tile_height,
    const uint32_t& tile_width,
    const uint32_t& face_height,
    const uint32_t& face_width,
    const uint32_t& face_num_x,
    const uint32_t& face_num_y,
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb,
    const IAGF<input_tensor_is_dram>& input_addr_gtor,
    const IAGF<index_tensor_is_dram>& index_addr_gtor,
    const IAGF<source_tensor_is_dram>& source_addr_gtor) {
    // for each tile along the scatter axis (Wt_input, or input_shape[scatter_axis]) in the input tensor...
    const uint32_t tile_width_mask = tile_width - 1;
    const uint32_t tile_height_mask = tile_height - 1;
    const uint32_t tile_width_divisor = __builtin_ctz(tile_width);
    const uint32_t tile_height_divisor = __builtin_ctz(tile_height);
    const uint32_t ht_pad = ((pad_scalar_offset + tile_height_mask) >> tile_height_divisor);
    const uint32_t residue_rows = (pad_scalar_offset & tile_height_mask);
    const uint32_t tile_hw = tile_height * tile_width;
    const uint32_t face_hw = face_height * face_width;
    for (uint32_t tile_input = 0; tile_input < Wt_input; ++tile_input) {
        cb_reserve_back(output_cb, ONE_TILE);
        // read an input tile + get fresh pointers
        read_nth_tile_in_Wt_tiles<input_tensor_is_dram>(input_addr_gtor, input_cb, Wt_input, ht_offset, tile_input);
        const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
        const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
        volatile tt_l1_ptr number_type* input_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
        volatile tt_l1_ptr number_type* output_l1_ptr =
            reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);

        // copy WIP input tile into output (memcpy?)
        for (uint32_t tile_input_inner = 0; tile_input_inner < tile_hw; ++tile_input_inner) {
            output_l1_ptr[tile_input_inner] = input_l1_ptr[tile_input_inner];
        }
        // ...scatter Wt_tiles (index_shape[scatter_axis]) tiles onto ONE_TILE from Wt_input
        for (uint32_t tile_index_w = 0; tile_index_w < Wt_index; ++tile_index_w) {
            // read index and source tiles + get fresh pointers
            read_nth_tile_in_Wt_tiles<index_tensor_is_dram>(
                index_addr_gtor, index_cb, Wt_index, ht_offset, tile_index_w);
            read_nth_tile_in_Wt_tiles<source_tensor_is_dram>(
                source_addr_gtor, source_cb, Wt_index, ht_offset, tile_index_w);
            const uint32_t index_l1_read_addr = get_read_ptr(index_cb);
            const uint32_t source_l1_read_addr = get_read_ptr(source_cb);
            volatile tt_l1_ptr index_type* index_l1_ptr =
                reinterpret_cast<volatile tt_l1_ptr index_type*>(index_l1_read_addr);
            volatile tt_l1_ptr number_type* source_l1_ptr =
                reinterpret_cast<volatile tt_l1_ptr number_type*>(source_l1_read_addr);
            // gut the tiles
            for (uint32_t face_x = 0; face_x < face_num_x; ++face_x) {
                for (uint32_t face_y = 0; face_y < face_num_y; ++face_y) {
                    for (uint32_t scalar_x = 0; scalar_x < face_width; ++scalar_x) {
                        const uint32_t width_scalar_index =
                            get_width_scalar_index(tile_index_w, face_x, scalar_x, tile_width, face_width);
                        // break sooner if the pointer went past logical width
                        if (width_scalar_index >= logical_index_width) {
                            break;
                        }
                        for (uint32_t scalar_y = 0; scalar_y < face_height; ++scalar_y) {
                            // get global coords + assert
                            // break sooner if the pointer went past logical height
                            if (counter == ht_pad - 1) {
                                const uint32_t height_scalar_index =
                                    get_height_scalar_index_inside_tile(face_y, scalar_y, face_height);
                                if (height_scalar_index >= residue_rows) {
                                    break;
                                }
                            }

                            // get scatter info
                            volatile index_type& index_value = tile_guts<index_type>(
                                index_l1_ptr, face_x, face_y, scalar_x, scalar_y, face_hw, face_width);
                            // check if index value targets currently chosen input tile (tile_input)
                            const uint32_t dest_tile_id_in_row = index_value >> tile_width_divisor;
                            ASSERT(dest_tile_id_in_row < Wt_input);
                            if (dest_tile_id_in_row != tile_input) {
                                continue;
                            }
                            volatile number_type& source_value = tile_guts<number_type>(
                                source_l1_ptr, face_x, face_y, scalar_x, scalar_y, face_hw, face_width);
                            const uint32_t x_index_in_tile = index_value & tile_width_mask;
                            const uint32_t dest_scalar_x =
                                (x_index_in_tile < face_width) ? x_index_in_tile : (x_index_in_tile - face_width);
                            const uint32_t dest_face_id_x = (x_index_in_tile < face_width) ? 0 : 1;

                            // scatter the value
                            tile_guts<number_type>(
                                output_l1_ptr, dest_face_id_x, face_y, dest_scalar_x, scalar_y, face_hw, face_width) =
                                source_value;
                        }
                    }
                }
            }
            // release index and source tiles
            cb_pop_front(index_cb, ONE_TILE);
            cb_pop_front(source_cb, ONE_TILE);
        }

        // release input tile + push resulting tile to the output
        cb_push_back(output_cb, ONE_TILE);
        cb_pop_front(input_cb, ONE_TILE);
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

    const uint32_t start_ht_id = get_arg_val<uint32_t>(0);
    const uint32_t ht_per_core = get_arg_val<uint32_t>(1);

    using input_std_type = std_type_t<get_dataformat(ctas.input_tensor_cb)>;
    using index_std_type = std_type_t<get_dataformat(ctas.index_tensor_cb)>;

    uint32_t counter = get_arg_val<uint32_t>(2);

    const uint32_t tile_height_mask = ctas.tile_height - 1;
    const uint32_t tile_height_divisor = __builtin_ctz(ctas.tile_height);
    constexpr uint32_t ht_pad = ((ctas.pad_scalar_offset + tile_height_mask) >> tile_height_divisor);

    for (uint32_t ht = start_ht_id; ht < start_ht_id + ht_per_core; ++ht) {
        if (counter == ht_pad) {
            counter = 0;
        }
        scatter_along_whole_axis<
            input_std_type,
            index_std_type,
            ctas.input_tensor_is_dram,
            ctas.index_tensor_is_dram,
            ctas.source_tensor_is_dram>(
            ht,
            ctas.Wt_input,
            ctas.logical_index_width,
            ctas.logical_index_height,
            ctas.Wt_index,
            counter,
            ctas.pad_scalar_offset,
            ctas.tile_height,
            ctas.tile_width,
            ctas.face_height,
            ctas.face_width,
            ctas.face_num_x,
            ctas.face_num_y,
            ctas.input_tensor_cb,
            ctas.index_tensor_cb,
            ctas.source_tensor_cb,
            ctas.output_tensor_cb,
            input_addr_gtor,
            index_addr_gtor,
            source_addr_gtor);
        ++counter;
    }
}

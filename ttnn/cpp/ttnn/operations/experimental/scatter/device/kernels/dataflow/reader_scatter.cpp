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

template <typename number_type, typename index_type>
FORCE_INLINE void scatter_Wt_src_tiles_from_src_as_per_index_onto_input_and_push_to_output(
    const uint32_t& h,
    const uint32_t& Wt_input,
    const uint32_t& logical_index_width,
    const uint32_t& logical_index_height,
    const uint32_t& Wt_index,
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb) {
    // working with Wt_input (== Wt_index) tiles at the same time - will implement a full algo with sorting soon.
    cb_wait_front(input_cb, Wt_input);
    cb_wait_front(index_cb, Wt_index);
    cb_wait_front(source_cb, Wt_index);

    cb_reserve_back(output_cb, Wt_input);
    const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
    const uint32_t index_l1_read_addr = get_read_ptr(index_cb);
    const uint32_t source_l1_read_addr = get_read_ptr(source_cb);
    const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
    volatile tt_l1_ptr number_type* input_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr index_type* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr index_type*>(index_l1_read_addr);
    volatile tt_l1_ptr number_type* source_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(source_l1_read_addr);
    volatile tt_l1_ptr number_type* output_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);

    // copy input to output first
    for (uint32_t i = 0; i < tt::constants::TILE_HW * Wt_input; ++i) {
        output_l1_ptr[i] = input_l1_ptr[i];
    }

    // constexpr DataFormat input_data_format = get_data_format(input_cb);
    // constexpr DataFormat index_data_format = get_data_format(index_cb);
    // constexpr DataFormat source_data_format = get_data_format(source_cb);

    // DPRINT << "h: " << h << ENDL();
    // DPRINT << "Wt_index: " << Wt_index << ENDL();
    // DPRINT << "logical_index_width: " << logical_index_width << ENDL();

    // scatter along Wt tiles
    for (uint32_t tile_id = 0; tile_id < Wt_index; ++tile_id) {
        for (uint32_t face_x = 0; face_x < 2; ++face_x) {
            for (uint32_t face_y = 0; face_y < 2; ++face_y) {
                for (uint32_t scalar_x = 0; scalar_x < 16; ++scalar_x) {
                    for (uint32_t scalar_y = 0; scalar_y < 16; ++scalar_y) {
                        const uint32_t width_scalar_index = get_width_scalar_index(tile_id, face_x, scalar_x);
                        const uint32_t height_scalar_index = get_height_scalar_index(h, face_y, scalar_y);
                        if (width_scalar_index >= logical_index_width || height_scalar_index >= logical_index_height) {
                            continue;
                        }

                        // get scatter info
                        volatile index_type& index_value =
                            tile_guts<index_type>(index_l1_ptr, face_x, face_y, scalar_x, scalar_y, tile_id);
                        if (index_value >= logical_index_width) {
                            continue;
                        }
                        volatile number_type& source_value =
                            tile_guts<number_type>(source_l1_ptr, face_x, face_y, scalar_x, scalar_y, tile_id);
                        const uint32_t dest_tile_id_in_row = index_value / 32;
                        const uint32_t x_index_in_tile = index_value % 32;
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
    cb_push_back(output_cb, Wt_input);
    cb_pop_front(input_cb, Wt_input);
    cb_pop_front(index_cb, Wt_index);
    cb_pop_front(source_cb, Wt_index);
}

// TODO(jbbieniekTT): stream-scatter after sorting (pt. 2 of the whole implementation) (issue no #)
template <typename number_type, typename index_type>
FORCE_INLINE void scatter_along_whole_axis(
    const uint32_t& Ht,
    const uint32_t& Wt_input,
    const uint32_t& Wt_index,
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb);

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

    using input_std_format = std_type_t<get_dataformat(ctas.input_tensor_cb)>;
    using index_std_format = std_type_t<get_dataformat(ctas.index_tensor_cb)>;
    // using source_std_format = std_type_t<get_dataformat(ctas.source_tensor_cb)>;

    // DPRINT << "::: " << start_ht_id << "::: " << ht_per_core << ENDL();

    for (uint32_t h = start_ht_id; h < start_ht_id + ht_per_core; ++h) {
        // first phase: read input/index/src
        read_wt_tiles<ctas.input_tensor_is_dram>(input_addr_gtor, ctas.input_tensor_cb, ctas.Wt_input, h);
        read_wt_tiles<ctas.index_tensor_is_dram>(index_addr_gtor, ctas.index_tensor_cb, ctas.Wt_index, h);
        read_wt_tiles<ctas.source_tensor_is_dram>(source_addr_gtor, ctas.source_tensor_cb, ctas.Wt_index, h);

        // second phase: copy input to output + scatter src onto output + push output
        scatter_Wt_src_tiles_from_src_as_per_index_onto_input_and_push_to_output<input_std_format, index_std_format>(
            h,
            ctas.Wt_input,
            ctas.logical_index_width,
            ctas.logical_index_height,
            ctas.Wt_index,
            ctas.input_tensor_cb,
            ctas.index_tensor_cb,
            ctas.source_tensor_cb,
            ctas.output_tensor_cb);
    }
}

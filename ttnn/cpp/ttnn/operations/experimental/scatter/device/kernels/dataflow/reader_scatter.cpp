// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

template <typename unsigned_type, typename number_type>
FORCE_INLINE void scatter_Wt_src_tiles_from_src_as_per_index_onto_input_and_push_to_output(
    const uint32_t& Ht,
    const uint32_t& Wt_input,
    const uint32_t& Wt_index,
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb) {
    // working with Wt_input (== Wt_index) tiles at the same time - will implement a full algo with sorting soon.
    cb_wait_front(input_cb, Wt_input);
    cb_wait_front(index_cb, Wt_index);

    cb_reserve_back(output_cb, Wt_input);
    const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
    const uint32_t index_l1_read_addr = get_read_ptr(index_cb);
    const uint32_t source_l1_read_addr = get_read_ptr(source_cb);
    const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
    volatile tt_l1_ptr number_type* input_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr unsigned_type* index_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr unsigned_type*>(index_l1_read_addr);
    volatile tt_l1_ptr number_type* source_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(source_l1_read_addr);
    volatile tt_l1_ptr number_type* output_l1_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);

    // copy input to output first
    for (uint32_t i = 0; i < tt::constants::TILE_HW * Wt_input; ++i) {
        output_l1_ptr[i] = input_l1_ptr[i];
    }

    for (uint32_t face_x = 0; face_x < 2; ++face_x) {
        for (uint32_t face_y = 0; face_y < 2; ++face_y) {
            for (uint32_t scalar_x = 0; scalar_x < 16; ++scalar_x) {
                for (uint32_t scalar_y = 0; scalar_y < 16; ++scalar_y) {
                    // get scatter info
                    volatile unsigned_type& index_value =
                        tile_guts<unsigned_type>(index_l1_ptr, face_x, face_y, scalar_x, scalar_y);
                    volatile number_type& source_value =
                        tile_guts<number_type>(source_l1_ptr, face_x, face_y, scalar_x, scalar_y);
                    const uint32_t dest_tile_id_in_row = index_value / 32;
                    const uint32_t x_index_in_tile = index_value % 32;
                    const uint32_t dest_scalar_x = (x_index_in_tile < 16) ? x_index_in_tile : (x_index_in_tile - 16);
                    const uint32_t dest_face_id_x = (x_index_in_tile < 16) ? 0 : 1;

                    // scatter the value
                    tile_guts<number_type>(
                        output_l1_ptr, dest_face_id_x, face_y, dest_scalar_x, scalar_y, dest_tile_id_in_row) =
                        source_value;
                }
            }
        }
    }
    cb_push_back(output_cb, Wt_input);
}

// TODO(jbbieniekTT): stream-scatter after sorting
template <typename unsigned_type, typename number_type>
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

    // TODO(jbbieniekTT): choose type in compile-time, if possible, to be used in all datum-related utility functions.
    const DataFormat input_data_format{get_dataformat(ctas.input_tensor_cb)};

    const auto input_addr_gtor{make_addr_gtor<ctas.input_tensor_is_dram>(ctas.input_tensor_cb, ctas.input_tensor_addr)};
    const auto index_addr_gtor{make_addr_gtor<ctas.index_tensor_is_dram>(ctas.index_tensor_cb, ctas.index_tensor_addr)};
    const auto source_addr_gtor{
        make_addr_gtor<ctas.source_tensor_is_dram>(ctas.source_tensor_cb, ctas.source_tensor_addr)};

    // TODO(jbbieniekTT): multi-core
    // for (uint32_t core_loop = 0; core_loop < ctas.core_loop_count; core_loop++) {
    // const uint32_t h = core_loop * total_number_of_cores +
    //                    get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

    // TODO(jbbieniekTT): multi-core
    // const uint32_t ht_offset = calculate_ht_offset_for_core(core_loop, ctas.total_number_of_cores,
    // ctas.compute_with_storage_grid_size_x);
    // const uint32_t ht_offset = 0;

    for (uint32_t h = 0; h < ctas.Ht; ++h) {
        // first phase: read input/index/src
        DPRINT << "READING EVERYTHING Wt_input = " << ctas.Wt_input << ENDL();
        read_wt_tiles<ctas.input_tensor_is_dram>(input_addr_gtor, ctas.input_tensor_cb, ctas.Wt_input, h);
        DPRINT << "INPUT TENSOR READ" << ENDL();
        read_wt_tiles<ctas.index_tensor_is_dram>(index_addr_gtor, ctas.index_tensor_cb, ctas.Wt_index, h);
        DPRINT << "INDEX TENSOR READ" << ENDL();
        read_wt_tiles<ctas.source_tensor_is_dram>(source_addr_gtor, ctas.source_tensor_cb, ctas.Wt_index, h);
        DPRINT << "SOURCE TENSDR READ" << ENDL();

        // second phase: copy input to output + scatter src onto output + push output
        scatter_Wt_src_tiles_from_src_as_per_index_onto_input_and_push_to_output<uint32_t, float>(
            ctas.Ht,
            ctas.Wt_input,
            ctas.Wt_index,
            ctas.input_tensor_cb,
            ctas.index_tensor_cb,
            ctas.source_tensor_cb,
            ctas.output_tensor_cb);
    }
}

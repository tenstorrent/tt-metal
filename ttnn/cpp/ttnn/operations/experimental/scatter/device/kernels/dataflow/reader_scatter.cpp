// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"
// #include "dataflow_api_addrgen.h"

// constraints:
// Wh_input < 9 :(
//

namespace {

template <bool is_dram>
FORCE_INLINE void read_wt_tiles(
    IAGF<is_dram> addr_gtor, const uint32_t& cb, const uint32_t& wt_num, const uint32_t& ht_offset) {
    for (uint32_t tile = 0; tile < wt_num; ++tile) {
        cb_reserve_back(cb, wt_num);
        const uint32_t l1_addr = get_write_ptr(cb);
        noc_async_read_tile(get_tile_id(tile, ht_offset), addr_gtor, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb, wt_num);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const auto input_addr_gtor{make_addr_gtor<ctas.input_tensor_is_dram>(ctas.input_tensor_cb, ctas.input_tensor_addr)};
    const auto index_addr_gtor{make_addr_gtor<ctas.index_tensor_is_dram>(ctas.index_tensor_cb, ctas.index_tensor_addr)};
    const auto src_addr_gtor{make_addr_gtor<ctas.src_tensor_is_dram>(ctas.src_tensor_cb, ctas.src_tensor_addr)};
    const auto output_addr_gtor{
        make_addr_gtor<ctas.output_tensor_is_dram>(ctas.output_tensor_cb, ctas.output_tensor_addr)};

    // TODO(jbbieniekTT): multi-core
    // for (uint32_t core_loop = 0; core_loop < ctas.core_loop_count; core_loop++) {
    // const uint32_t h = core_loop * total_number_of_cores +
    //                    get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

    // TODO(jbbieniekTT): multi-core
    // const uint32_t ht_offset = calculate_ht_offset_for_core(core_loop, ctas.total_number_of_cores,
    // ctas.compute_with_storage_grid_size_x);
    const uint32_t ht_offset = 0;

    // first phase: read input/index/src
    read_wt_tiles(ctas.input_tensor_cb, input_addr_gtor, ctas.Wt_input, ht_offset);
    DPRINT << "INPUT TENSOR READ";
    read_wt_tiles(ctas.index_tensor_cb, index_addr_gtor, ctas.Wt_index, ht_offset);
    DPRINT << "INDEX TENSOR READ";
    read_wt_tiles(ctas.src_tensor_cb, src_addr_gtor, ctas.Wt_index, ht_offset);
    DPRINT << "SRC TENSDR READ";

    //
    // read_input_wt_tiles();
    // read_index_wt_tiles();
    // read_src_wt_tiles();

    // second phase
    // scatter_src_per_index_onto_input()

    // third phase
    // move_input_to_output_and_push();
    // }
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

namespace {

template <bool is_dram>
FORCE_INLINE void write_wt_tiles(
    IAGF<is_dram> addr_gtor, const uint32_t& cb, const uint32_t Wt_input, const uint32_t& ht_offset = 0) {
    for (uint32_t tile_num_in_row = 0; tile_num_in_row < Wt_input; ++tile_num_in_row) {
        cb_wait_front(cb, ONE_TILE);
        const uint32_t l1_addr = get_read_ptr(cb);
        noc_async_write_tile(ht_offset * Wt_input + tile_num_in_row, addr_gtor, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb, ONE_TILE);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const auto output_addr_gtor{
        make_addr_gtor<ctas.output_tensor_is_dram>(ctas.output_tensor_cb, ctas.output_tensor_addr)};

    const uint32_t start_ht_id = get_arg_val<uint32_t>(0);
    const uint32_t ht_per_core = get_arg_val<uint32_t>(1);

    for (uint32_t h = start_ht_id; h < start_ht_id + ht_per_core; ++h) {
        // simply read the output_tensor_cb and write to the NoC
        // this writes tile-by-tile as soon as a single tile becomes available in the output CB
        write_wt_tiles<ctas.output_tensor_is_dram>(output_addr_gtor, ctas.output_tensor_cb, ctas.Wt_input, h);
    }
}

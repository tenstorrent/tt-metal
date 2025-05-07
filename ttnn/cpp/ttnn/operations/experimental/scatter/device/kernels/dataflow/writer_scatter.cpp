// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

namespace {

template <bool is_dram>
FORCE_INLINE void write_wt_tiles(
    IAGF<is_dram> addr_gtor, const uint32_t& cb, const uint32_t Wt_input, const uint32_t& ht_offset) {
    for (uint32_t tile = 0; tile < Wt_input; ++tile) {
        cb_wait_front(cb, ONE_TILE);
        // cb_wait_front(cb, Wt_input);
        const uint32_t l1_addr = get_read_ptr(cb);
        noc_async_write_tile(get_tile_id(), addr_gtor, l1_addr);
        noc_async_write_barrier();
        // cb_pop_back(cb, Wt_input);
        cb_pop_back(cb, ONE_TILE);
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
    const uint32_t ht_offset = 0;

    write_wt_tiles(output_addr_gtor, ctas.output_tensor_cb, ctas.Wt_input, ht_offset);
}

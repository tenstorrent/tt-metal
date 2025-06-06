// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

namespace detail {
template <class AddrGenT>
void noc_async_read(const uint32_t i, const AddrGenT& s, const uint32_t l1_write_addr) {
#ifdef ROWMAJOR
    const uint64_t src_noc_addr = get_noc_addr(i, s);
    ::noc_async_read(src_noc_addr, l1_write_addr, s.page_size);
#else
    noc_async_read_tile(i, s, l1_write_addr);

#endif
}
}  // namespace detail

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const DataFormat data_format = get_dataformat(cb_id_in0);

#ifdef ROWMAJOR
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);
    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_bytes};  //, .data_format = data_format};
#else
    const uint32_t page_bytes = get_tile_size(cb_id_in0);
    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_bytes};  //, .data_format = data_format};
#endif

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        detail::noc_async_read(i, s, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
    }
}

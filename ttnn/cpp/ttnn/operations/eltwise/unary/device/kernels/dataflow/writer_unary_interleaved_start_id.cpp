// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

namespace detail {
template <bool DstIsDram>
void noc_async_write(const uint32_t i, const InterleavedAddrGenFast<DstIsDram>& s, const uint32_t l1_read_addr) {
#ifdef ROWMAJOR
    const uint64_t dst_noc_addr = s.get_noc_addr(i);
    ::noc_async_write(l1_read_addr, dst_noc_addr, s.page_size);
#else
    noc_async_write_tile(i, s, l1_read_addr);

#endif
}

}  // namespace detail

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const DataFormat data_format = get_dataformat(cb_id_out);

#ifdef ROWMAJOR
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);
#else
    const uint32_t page_bytes = get_tile_size(cb_id_in0);
#endif

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = page_bytes, .data_format = data_format};

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);
#else

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        detail::noc_async_write(i, s, l1_read_addr);

        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
#endif
}

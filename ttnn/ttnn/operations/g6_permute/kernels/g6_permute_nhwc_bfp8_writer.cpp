// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute writer (nhwc_tile_bfp8_transpose): store_group.
// Slot c of cb_output_tiles is output tile (n, 32 * ct + c, ht, wt). Only the c_live slots that hold
// a channel are written (C not a multiple of 32 leaves the rest as input padding columns). One
// flush per group before the slots are released, one write barrier at the end.
// Raw dataflow: no kernel_lib helper writes CB tiles to TensorAccessor pages.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_nhwc_bfp8_common.hpp"

using namespace g6_permute_nhwc;

constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tensor_c = get_compile_time_arg_val(3);
constexpr uint32_t tensor_ht = get_compile_time_arg_val(4);
constexpr uint32_t tensor_wt = get_compile_time_arg_val(5);
constexpr uint32_t tensor_ct = get_compile_time_arg_val(6);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(7);
// Args 1, 2 (tensor_h, tensor_w) belong to the reader; the layout is shared.
constexpr auto output_args = TensorAccessorArgs<8>();

using Cursor = GroupCursor<tensor_ht, tensor_wt, tensor_ct>;

// Consecutive c of one (n, ht, wt) are this many output pages apart.
constexpr uint32_t c_page_stride = tensor_ht * tensor_wt;

template <typename Accessor>
FORCE_INLINE void store_group(const Accessor& output, const Cursor& g) {
    cb_wait_front(cb_output_tiles, GROUP_TILES);
    const uint32_t c_live = live_extent(tensor_c, g.ct);
    uint32_t page = ((g.n * tensor_c + GROUP_TILES * g.ct) * tensor_ht + g.ht) * tensor_wt + g.wt;
    uint32_t slot = get_read_ptr(cb_output_tiles);
    for (uint32_t c = 0; c < c_live; ++c) {
        noc_async_write(slot, output.get_noc_addr(page), out_page_bytes);
        page += c_page_stride;
        slot += out_page_bytes;
    }
    // The slots are the writes' source: they must have left L1 before compute may refill them.
    noc_async_writes_flushed();
    cb_pop_front(cb_output_tiles, GROUP_TILES);
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(0);
    Cursor cursor{
        get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);

    for (uint32_t g = 0; g < num_groups; ++g) {
        DeviceZoneScopedN("PERM-NHWC-WR-GROUP");
        store_group(output, cursor);
        cursor.advance();
    }
    // The next program starts right after this one: no write may still be in flight.
    noc_async_write_barrier();
}

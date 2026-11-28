// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_sharded_to_interleaved writer, one instance per write lane on every core of the input's
// shard grid. The core's shard is resident in L1 behind cb_input_shard (a CB bound to the input's
// own buffer), so store_block writes each tile of the core's block straight from its shard slot to
// its interleaved output page: one NoC write per tile, no NoC read, one barrier per kernel.
//
// Raw dataflow API, no kernel_lib helper: the library has no writer from a resident L1 source to
// NoC-addressed interleaved pages (eltwise copy goes through unpack/pack and would regroup bfp8
// exponents; the local-copy helpers target local L1 only).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

// Slots keep the nominal shard stride even in a short last shard, so the row stride is
// shard_tile_cols pages, never block_tile_cols. Interleaved pages lie in different banks, so each
// tile is its own write; page_size fits one NoC packet (host-asserted).
template <uint32_t page_size, typename Accessor>
FORCE_INLINE void store_block(
    const Accessor& output,
    uint32_t first_slot_addr,
    uint32_t first_page,
    uint32_t block_tile_rows,
    uint32_t block_tile_cols,
    uint32_t shard_row_stride_bytes,
    uint32_t tensor_tile_cols) {
    uint32_t row_slot_addr = first_slot_addr;
    uint32_t row_page = first_page;
    for (uint32_t r = 0; r < block_tile_rows; ++r) {
        uint32_t slot_addr = row_slot_addr;
        for (uint32_t c = 0; c < block_tile_cols; ++c) {
            noc_async_write<page_size>(slot_addr, output.get_noc_addr(row_page + c), page_size);
            slot_addr += page_size;
        }
        row_slot_addr += shard_row_stride_bytes;
        row_page += tensor_tile_cols;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_shard = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tensor_tile_cols = get_common_arg_val<uint32_t>(1);
    const uint32_t shard_tile_cols = get_common_arg_val<uint32_t>(2);

    // num_blocks_this_core is 0 on a grid core without a shard (or without a lane piece).
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(0);
    const uint32_t block_first_slot = get_arg_val<uint32_t>(1);
    const uint32_t block_first_page = get_arg_val<uint32_t>(2);
    const uint32_t block_tile_rows = get_arg_val<uint32_t>(3);
    const uint32_t block_tile_cols = get_arg_val<uint32_t>(4);

    const auto output = TensorAccessor(output_args, output_addr, page_size);
    // cb_input_shard is never waited on or popped: the shard is resident before launch.
    const uint32_t shard_base_addr = get_read_ptr(cb_input_shard);

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        DeviceZoneScopedN("S2I-STORE-BLOCK");
        store_block<page_size>(
            output,
            shard_base_addr + block_first_slot * page_size,
            block_first_page,
            block_tile_rows,
            block_tile_cols,
            shard_tile_cols * page_size,
            tensor_tile_cols);
    }

    {
        DeviceZoneScopedN("S2I-WRITE-BARRIER");
        noc_async_write_barrier();
    }
}

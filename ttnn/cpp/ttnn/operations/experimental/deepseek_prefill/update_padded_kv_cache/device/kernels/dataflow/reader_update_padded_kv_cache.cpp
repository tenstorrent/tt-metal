// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2.0 fork of ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp
// for the deepseek_prefill update_padded_kv_cache op. The kv_cache version is still on Device 1.x; this fork lets
// deepseek_prefill flip cleanly to D2.0 without dragging the entire kv_cache op along.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;
    CircularBuffer cb_in0(cb_id_in0);

#ifdef INPUT_SHARDED
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
#else
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(src_args, src_addr);
    Noc noc;
    // Page bytes must come from the CB's configured page size, NOT get_tile_size(): the op runs in
    // ROW_MAJOR too, where a page is one token row (head_dim * element bytes) while get_tile_size()
    // still reports the dtype's 32x32 tile size. Reading tile-size bytes into a row-major page slot
    // over-reads the source page and overruns the CB in L1 (it scribbled the writer's metadata
    // scratch, which sits directly after this CB, corrupting the on-device slot_idx /
    // kv_actual_global read). The writer derives its page bytes the same way.
    const uint32_t src_page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_in0.reserve_back(onetile);
        noc.async_read(s, cb_in0, src_page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
#endif
}

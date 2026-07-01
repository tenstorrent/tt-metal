// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t mask_w = get_arg_val<uint32_t>(3);
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t scaler = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t BATCH = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);

    constexpr uint32_t cb_id_in2 = 2;
    CircularBuffer cb_in2_obj(cb_id_in2);
    generate_mm_scaler(cb_in2_obj, scaler);

    constexpr uint32_t cb_id_mask_w = 3;
#ifdef DO_MASK_W
    CircularBuffer cb_mask_w_obj(cb_id_mask_w);
    generate_mask_w(cb_mask_w_obj, mask_w);
#endif

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer cb_in0_obj(cb_id_in0);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);

    // Batched reads: issue BATCH async_reads into distinct slots of one
    // multi-tile CB reservation, then a SINGLE barrier per batch. This overlaps
    // the NoC round-trips of the batch instead of serializing one barrier per
    // single-tile read. The original op was reader(BRISC)-bound (BRISC duration
    // == kernel duration) and latency-bound on per-tile NoC round-trips; the
    // factory sizes cb_in0 to 2*BATCH so the reader double-buffers (runs a batch
    // ahead of compute). BATCH is a compile-time arg (4, capped by Wt).
    const uint32_t end_id = start_id + num_tiles;
    uint32_t i = start_id;
    for (; i + BATCH <= end_id; i += BATCH) {
        cb_in0_obj.reserve_back(BATCH);
        for (uint32_t j = 0; j < BATCH; ++j) {
            noc.async_read(s, cb_in0_obj, in0_tile_bytes, {.page_id = i + j}, {.offset_bytes = j * in0_tile_bytes});
        }
        noc.async_read_barrier();
        cb_in0_obj.push_back(BATCH);
    }
    // Remainder (num_tiles not a multiple of BATCH).
    for (; i < end_id; ++i) {
        cb_in0_obj.reserve_back(onetile);
        noc.async_read(s, cb_in0_obj, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0_obj.push_back(onetile);
    }
}

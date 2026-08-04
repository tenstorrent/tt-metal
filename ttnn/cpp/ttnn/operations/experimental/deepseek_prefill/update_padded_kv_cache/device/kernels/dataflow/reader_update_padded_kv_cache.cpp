// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2.0 fork of kv_cache/.../reader_fill_cache_interleaved_start_id.cpp, so deepseek_prefill can move to
// D2.0 without dragging the (still Device 1.x) kv_cache op along.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);  // per-core; buffers arrive as Buffer* -> addresses
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(2);

    const uint32_t linear_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t linear_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);  // page-rows THIS chip writes (fine stripe)
    const uint32_t input_Ht = get_common_arg_val<uint32_t>(3);       // input page-rows (SP chip's coarse window)
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(4);
    const uint32_t tp_factor = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt = get_common_arg_val<uint32_t>(6);
    // Args 0-6 are structural per cached program; this one is patched per call on cache hits by
    // override_runtime_arguments (the writer's contract too) -- which rows this chip owns depends on the
    // chunk start, not just its mesh position.
    const uint32_t kv_actual_global = get_common_arg_val<uint32_t>(7);

    constexpr uint32_t tile_height = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_in0 = 0;
    CircularBuffer cb_in0(cb_id_in0);

#ifdef INPUT_SHARDED
    cb_in0.reserve_back(num_pages);
    cb_in0.push_back(num_pages);
#else
    // Source-row mapping. The input is TP-REPLICATED: SP chip s holds its whole coarse window of input_Ht
    // rows in the writer's ROTATED order, and this chip (linear rank s*tp + t) keeps only the
    // chunk_local_t rows whose GLOBAL position is its own. A contiguous t*chunk_local_t slice is right
    // only when the coarse start is stripe-aligned; a sub-stripe start puts the fine ownership boundaries
    // INSIDE the slice, shearing it so each chip persists its neighbour's rows. Inverting the rotation
    // over this chip's (contiguous) destination rows gives src_block(j) = base + j, plus `jump` once
    // j + o crosses a stripe: one run, or two when this chip straddles the coarse slab boundary.
    // o / u0: fine and coarse start offsets, nonzero only on the single chip / single SP group owning
    // each boundary (the fine test is the writer's own expression, so o == its update_idxt % stripe).
    // base: + input_Ht keeps the numerator positive (u0 < input_Ht), sum < 2*input_Ht, so one mod.
    // jump: skip the other TP chips' stripes, since input_Ht == chunk_local_t*tp_factor; 0 at tp=1.
    const uint32_t kv_t = kv_actual_global / tile_height;
    const uint32_t o = (linear_coord == (kv_t / chunk_local_t) % linear_factor) ? kv_t % chunk_local_t : 0;
    const uint32_t u0 = (linear_coord / tp_factor == (kv_t / input_Ht) % sp_factor) ? kv_t % input_Ht : 0;
    const uint32_t base = ((linear_coord % tp_factor) * chunk_local_t + o + input_Ht - u0) % input_Ht;
    const uint32_t jump = chunk_local_t * (tp_factor - 1);

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(src_args, src_addr);
    Noc noc;
    // CB page size, NOT get_tile_size(): in ROW_MAJOR a page is one token row, and a tile-size read
    // overruns the CB into the writer's metadata scratch. The writer derives its page bytes the same way.
    const uint32_t src_page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // Destination order (block j, then its Wt pages) -- the writer consumes the CB in lockstep.
    const uint32_t n_blocks = num_pages / Wt;
    for (uint32_t k = 0; k < n_blocks; ++k) {
        const uint32_t j = core_blocks_written + k;
        const uint32_t src_page0 = (base + j + (j + o >= chunk_local_t ? jump : 0)) * Wt;
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_in0.reserve_back(onetile);
            noc.async_read(s, cb_in0, src_page_bytes, {.page_id = src_page0 + w}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
        }
    }
#endif
}

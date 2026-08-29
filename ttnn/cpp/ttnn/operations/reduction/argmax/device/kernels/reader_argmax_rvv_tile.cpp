// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// reader_argmax_rvv_tile.cpp — dataflow side of the RVV TILE-layout argmax.
//
// Streams the input tiles of each 32-row tile-row into the (double-buffered)
// input CB in chunks, then collects the per-row (index, maxval) results the
// compute kernel pushes and writes them to the output tensor(s).
//
// The input CB is treated as a ring of pages addressed by GLOBAL page index
// (slot = t % num_pages) on BOTH sides, so chunk batches may wrap mid-batch
// without any linear-placement assumption. NOC reads are issued per chunk
// with batched barriers — the scan on the compute side overlaps the next
// chunk's staging.
// =============================================================================

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "internal/tt-1xx/risc_common.h"  // invalidate_l1_cache()

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_res_idx = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res_val = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stage_idx = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stage_val = get_compile_time_arg_val(4);
    constexpr uint32_t src_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t chunk_pages = get_compile_time_arg_val(6);
    constexpr uint32_t in_cb_pages = get_compile_time_arg_val(7);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t logical_height = get_compile_time_arg_val(10);
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(11);
    constexpr uint32_t out_page_elems = get_compile_time_arg_val(12);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t val_page_size = get_compile_time_arg_val(14);
    constexpr bool has_maxval = (bool)get_compile_time_arg_val(15);
    constexpr uint32_t num_c_time_args = 16;

    // Base addresses are positional RUNTIME args by design (repo idiom, cf.
    // reader_argmax_tile_layout*.cpp): TensorAccessorArgs carry only the
    // compile-time banking/layout — TensorAccessor requires the base address
    // at construction so program-cache rebinds can update it per allocation.
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t val_base_addr = has_maxval ? get_arg_val<uint32_t>(2) : 0;

    constexpr auto s_src_args = TensorAccessorArgs<num_c_time_args>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();
    const auto s_src = TensorAccessor(s_src_args, src_base_addr, src_page_size);
    const auto s_dst = TensorAccessor(s_dst_args, dst_base_addr, dst_page_size);

    // The maxval accessor args are appended only when the tensor is present.
    constexpr auto s_val_args = TensorAccessorArgs<s_dst_args.next_compile_time_args_offset()>();
    const auto s_val = TensorAccessor(s_val_args, val_base_addr, val_page_size);

    Noc noc;
    CircularBuffer in_cb(cb_in);
    CircularBuffer res_idx_cb(cb_res_idx);
    CircularBuffer res_val_cb(cb_res_val);
    CircularBuffer stage_idx_cb(cb_stage_idx);
    CircularBuffer stage_val_cb(cb_stage_val);

    // Input CB ring base (write pointer sits at base before any push). The ring
    // is addressed by GLOBAL page index, so the base is captured ONCE here: the
    // CB's own write pointer advances on every push_back and cannot be used as
    // the NoC destination, hence a CoreLocalMem over the fixed base + offset.
    const CoreLocalMem<uint32_t> in_ring(in_cb.get_write_ptr());

    // Staging buffers for output pages (plain L1 scratch, no FIFO semantics).
    const CoreLocalMem<uint32_t> stage_idx(stage_idx_cb.get_write_ptr());
    const CoreLocalMem<uint16_t> stage_val(stage_val_cb.get_write_ptr());

    uint32_t t_global = 0;   // global input page counter (matches compute side)
    uint32_t collected = 0;  // elements accumulated toward the current output page
    uint32_t out_page_id = 0;

    for (uint32_t outer = 0; outer < outer_dim_units; outer++) {
        for (uint32_t i = 0; i < h_tiles; i++) {
            const uint32_t row_base = i * 32u;
            const uint32_t units = (logical_height - row_base < 32u) ? (logical_height - row_base) : 32u;
            const uint32_t tile_row_first = (outer * h_tiles + i) * w_tiles;

            uint32_t done = 0;
            while (done < w_tiles) {
                const uint32_t chunk = (w_tiles - done < chunk_pages) ? (w_tiles - done) : chunk_pages;
                in_cb.reserve_back(chunk);
                for (uint32_t k = 0; k < chunk; k++) {
                    const uint32_t slot = (t_global + k) % in_cb_pages;
                    noc.async_read(
                        s_src,
                        in_ring,
                        src_page_size,
                        {.page_id = tile_row_first + done + k},
                        {.offset_bytes = slot * src_page_size});
                    if ((k & 31u) == 31u) {
                        // Bound the outstanding NOC read count within a chunk.
                        noc.async_read_barrier();
                    }
                }
                noc.async_read_barrier();
                in_cb.push_back(chunk);
                t_global += chunk;
                done += chunk;
            }

            // Collect this tile-row pass's results from the compute kernel.
            res_idx_cb.wait_front(1);
            res_val_cb.wait_front(1);
            invalidate_l1_cache();
            const CoreLocalMem<volatile uint32_t> ip(res_idx_cb.get_read_ptr());
            const CoreLocalMem<volatile uint16_t> vp(res_val_cb.get_read_ptr());
            for (uint32_t r = 0; r < units; r++) {
                stage_idx[collected] = ip[r];
                if constexpr (has_maxval) {
                    stage_val[collected] = vp[r];
                }
                collected++;
                if (collected == out_page_elems) {
                    noc.async_write(stage_idx, s_dst, dst_page_size, {}, {.page_id = out_page_id});
                    if constexpr (has_maxval) {
                        noc.async_write(stage_val, s_val, val_page_size, {}, {.page_id = out_page_id});
                    }
                    noc.async_write_barrier();
                    collected = 0;
                    out_page_id++;
                }
            }
            res_idx_cb.pop_front(1);
            res_val_cb.pop_front(1);
        }
    }
}

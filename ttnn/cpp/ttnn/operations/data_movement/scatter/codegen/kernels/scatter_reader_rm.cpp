// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter reader – ROW_MAJOR path (NCRISC).
// Operates on flat sticks: output[index[i]] = src[i].
// No tile math — just flat element-level scatter on RM sticks.
//
// Protocol (dual-RISC):
//   Writer (BRISC): loads input stick→cb_input, writes cb_output→DRAM
//   Reader (NCRISC): waits cb_input, copies to cb_output, scatter, pushes
//
// Chunked index/src streaming (large-stick support):
//   The output stick stays fully resident in L1 (any index[i] can target any
//   output position), but the index and src sticks are streamed in fixed-size
//   chunks so their L1 footprint is bounded regardless of stick length. Each
//   chunk is a *sub-range of the single per-stick DRAM page*, read via a
//   page-relative Noc::async_read ({.page_id, .offset_bytes}) — never spanning
//   the page boundary, so the interleaved bank mapping is never crossed. chunk_elems is
//   chosen (by the builder) so chunk_elems*elem_size is a multiple of the 32B
//   NOC alignment for every dtype, keeping every chunk source/dest 32B aligned.
//   When the whole stick fits one chunk this collapses to the original 1-read
//   path (byte-identical behavior).

#include "scatter_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_stick = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);
    // Args 4..20 are the fixed-width logical page-map descriptor.

    // Compile-time args
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src = get_compile_time_arg_val(3);
    constexpr uint32_t input_stick_elems = get_compile_time_arg_val(4);
    constexpr uint32_t index_stick_elems = get_compile_time_arg_val(5);
    // Element sizes in bytes — passed explicitly because get_tile_size()
    // returns L1 words (not bytes), causing division to zero.
    constexpr uint32_t output_df_size = get_compile_time_arg_val(6);
    constexpr uint32_t index_df_size = get_compile_time_arg_val(7);
    constexpr uint32_t src_df_size = get_compile_time_arg_val(8);
    // Actual page sizes in bytes — passed explicitly because get_tile_size()
    // returns L1 words (not bytes) for RM CBs.
    constexpr uint32_t index_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t src_page_bytes = get_compile_time_arg_val(10);
    // Chunk size (elements) for streaming index/src. chunk_elems*elem_size is a
    // multiple of the 32B NOC alignment for every supported dtype (builder
    // enforces chunk_elems % 32 == 0), so every chunk read is 32B aligned.
    constexpr uint32_t chunk_elems = get_compile_time_arg_val(11);
    constexpr auto index_ta_args = TensorAccessorArgs<12>();
    constexpr auto src_ta_args = TensorAccessorArgs<index_ta_args.next_compile_time_args_offset()>();
    constexpr uint32_t reduction_mode = get_named_compile_time_arg_val("reduction_mode");
    constexpr uint32_t value_kind = get_named_compile_time_arg_val("value_kind");
    // scatter_reduce_value has no bfloat16 (kind 5) arm; a bf16 reduce belongs to
    // scatter_reader_bf16_reduce_rm.cpp, and here it would drop every update.
    static_assert(reduction_mode == 0 || value_kind != 5, "bfloat16 reduction is not served by this reader");

    constexpr uint32_t one_page = 1;

    const auto index_accessor = TensorAccessor(index_ta_args, index_addr, index_page_bytes);
    const auto src_accessor = TensorAccessor(src_ta_args, src_addr, src_page_bytes);

    Noc noc;
    CircularBuffer in_cb(cb_input);
    CircularBuffer out_cb(cb_output);
    CircularBuffer index_cb(cb_index);
    CircularBuffer src_cb(cb_src);
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    // The index/src CBs are single-owner L1 scratch (only this reader touches
    // them). Reserve once and reuse the same buffer for every chunk of every
    // stick — no per-chunk push/pop handshake is needed.
    index_cb.reserve_back(one_page);
    src_cb.reserve_back(one_page);
    const uint32_t idx_l1 = index_cb.get_write_ptr();
    const uint32_t src_l1 = src_cb.get_write_ptr();

    for (uint32_t s = 0; s < num_sticks; s++) {
        // Wait for input stick from writer
        in_cb.wait_front(one_page);
        out_cb.reserve_back(one_page);

        // Copy input → output via NOC L1→L1 (hardware DMA, much faster than
        // volatile loop). Must complete before any scatter write below.
        const uint32_t in_l1 = in_cb.get_read_ptr();
        const uint32_t out_l1 = out_cb.get_write_ptr();
        constexpr uint32_t stick_bytes = input_stick_elems * output_df_size;
        noc.async_read(
            self_ep, out_cb, stick_bytes, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = in_l1}, {.offset_bytes = 0});
        noc.async_read_barrier();

        const uint32_t input_stick_id = start_stick + s;
        uint32_t source_stick_id = 0;
        const bool has_src_data = map_scatter_input_page<4>(input_stick_id, source_stick_id);

        // Stream index+src in chunks that are sub-ranges of the single per-stick
        // page. base advances by chunk_elems every iteration (never underflows),
        // so the loop always terminates. Pages outside the compact update prefix
        // deliberately skip all reads and preserve the copied input unchanged.
        for (uint32_t base = 0; has_src_data && base < index_stick_elems; base += chunk_elems) {
            uint32_t this_chunk = chunk_elems;
            if (base + this_chunk > index_stick_elems) {
                this_chunk = index_stick_elems - base;
            }
            // 32B-aligned byte offsets (chunk_elems*elem_size % 32 == 0); the
            // final partial chunk reads exactly this_chunk*elem_size bytes
            // (unaligned size is fine — only addresses require 32B alignment)
            // and never crosses the page boundary.
            const uint32_t idx_off = base * index_df_size;
            const uint32_t src_off = base * src_df_size;
            noc.async_read(
                index_accessor,
                index_cb,
                this_chunk * index_df_size,
                {.page_id = source_stick_id, .offset_bytes = idx_off},
                {.offset_bytes = 0});
            noc.async_read(
                src_accessor,
                src_cb,
                this_chunk * src_df_size,
                {.page_id = source_stick_id, .offset_bytes = src_off},
                {.offset_bytes = 0});
            noc.async_read_barrier();

            // Flat scatter for this chunk: output[index[base+i]] = src[base+i].
            // idx_l1[i] and src_l1[i] both hold element (base+i).
            for (uint32_t i = 0; i < this_chunk; i++) {
                const uint32_t dest_idx = get_value_from_tile(idx_l1, i, index_df_size);
                if (dest_idx < input_stick_elems) {
                    const uint32_t val = get_value_from_tile(src_l1, i, src_df_size);
                    const uint32_t output_value = scatter_reduce_value(
                        get_value_from_tile(out_l1, dest_idx, output_df_size), val, reduction_mode, value_kind);
                    write_value_to_tile(out_l1, dest_idx, output_df_size, output_value);
                }
            }
        }

        in_cb.pop_front(one_page);
        out_cb.push_back(one_page);
    }
}

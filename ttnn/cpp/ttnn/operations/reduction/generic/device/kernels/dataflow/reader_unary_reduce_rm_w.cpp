// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reduce_rm_dataflow_common.hpp"

// Packs rm_rows_per_tile logical rows per staged page per W chunk. Per Ht chunk block: push order is
// (W chunk × slab) to match reduce_rm_w tilize order before ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC).
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page = get_arg_val<uint32_t>(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr uint32_t W_logical = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t padding_identity_bits = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t wt_tiles_per_chunk = get_compile_time_arg_val(5);
    constexpr uint32_t rm_rows_per_tile = get_compile_time_arg_val(6);
    constexpr uint32_t ht_tiles_per_chunk = get_compile_time_arg_val(7);
    constexpr auto tensor_args = TensorAccessorArgs<8>();

    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    const uint32_t scaler_valid_for_reduce = []() -> uint32_t {
        if constexpr (REDUCE_OP == ckernel::PoolType::SUM) {
            return tt::constants::TILE_WIDTH;
        }
        return (W_logical < tt::constants::TILE_WIDTH) ? W_logical : tt::constants::TILE_WIDTH;
    }();
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_scaler, REDUCE_OP, REDUCE_DIM>(scaler_f, scaler_valid_for_reduce);

    constexpr uint32_t cb_id_rm = tt::CBIndex::c_24;
    constexpr uint32_t cb_id_clear_value = tt::CBIndex::c_4;
    constexpr uint32_t onepage = 1;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_rm).fifo_page_size;
    const uint32_t valid_row_bytes = W_logical * elem_bytes;

    experimental::CircularBuffer cb_clear_value(cb_id_clear_value);
    const uint32_t clear_template_bytes = get_tile_size(cb_id_clear_value);
    rm_fill_buffer_with_identity_pattern(
        cb_clear_value.get_write_ptr(), clear_template_bytes, elem_bytes, padding_identity_bits);
    cb_clear_value.push_back(onepage);

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_rm(cb_id_rm);
    experimental::UnicastEndpoint self_ep;
    const auto clear_template_src = experimental::local_addr(cb_clear_value.get_read_ptr(), noc.get_noc_id());

    const uint32_t Ht_reader = (num_pages + rm_rows_per_tile - 1) / rm_rows_per_tile;
    uint32_t rows_remaining = num_pages;
    uint32_t packed_row_base = start_page;

    for (uint32_t ht_base = 0; ht_base < Ht_reader; ht_base += ht_tiles_per_chunk) {
        const uint32_t ht_in_chunk =
            (ht_base + ht_tiles_per_chunk < Ht_reader) ? ht_tiles_per_chunk : (Ht_reader - ht_base);

        uint32_t slab_first_page[8];
        uint32_t slab_rows_in_pack[8];
        uint32_t rows_left_for_slabs = rows_remaining;
        uint32_t page_cursor = packed_row_base;
        uint32_t total_rows_this_block = 0;

        for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
            const uint32_t rip = rows_left_for_slabs < rm_rows_per_tile ? rows_left_for_slabs : rm_rows_per_tile;
            slab_first_page[hti] = page_cursor;
            slab_rows_in_pack[hti] = rip;
            page_cursor += rip;
            rows_left_for_slabs -= rip;
            total_rows_this_block += rip;
        }

        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += wt_tiles_per_chunk) {
            const uint32_t wt_in_chunk = (wt_base + wt_tiles_per_chunk < Wt) ? wt_tiles_per_chunk : (Wt - wt_base);
            const uint32_t chunk_bytes = wt_in_chunk * tt::constants::TILE_WIDTH * elem_bytes;
            const uint32_t row_chunk_start_bytes = wt_base * tt::constants::TILE_WIDTH * elem_bytes;
            const uint32_t valid_bytes_this_chunk = (row_chunk_start_bytes >= valid_row_bytes)
                                                        ? 0
                                                        : ((valid_row_bytes - row_chunk_start_bytes) < chunk_bytes
                                                               ? (valid_row_bytes - row_chunk_start_bytes)
                                                               : chunk_bytes);

            for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
                const uint32_t rows_in_pack = slab_rows_in_pack[hti];
                const uint32_t slab_page0 = slab_first_page[hti];

                cb_rm.reserve_back(onepage);
                uint32_t pad_offset = 0;
                while (pad_offset < page_bytes) {
                    const uint32_t copy_bytes = (page_bytes - pad_offset) < clear_template_bytes
                                                    ? (page_bytes - pad_offset)
                                                    : clear_template_bytes;
                    noc.async_read(self_ep, cb_rm, copy_bytes, clear_template_src, {.offset_bytes = pad_offset});
                    pad_offset += copy_bytes;
                }

                for (uint32_t r = 0; r < rows_in_pack; ++r) {
                    if (valid_bytes_this_chunk > 0) {
                        noc.async_read(
                            tensor_accessor,
                            cb_rm,
                            valid_bytes_this_chunk,
                            {.page_id = slab_page0 + r, .offset_bytes = row_chunk_start_bytes},
                            {.offset_bytes = r * chunk_bytes});
                    }
                }
                noc.async_read_barrier();
                cb_rm.push_back(onepage);
            }
        }

        packed_row_base += total_rows_this_block;
        rows_remaining -= total_rows_this_block;
    }
}

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

// Row-major W chunks: one cb_rm page per (logical row, W-chunk). Matches compute tilize_block(..., wt_in_chunk, ...).
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
    constexpr auto tensor_args = TensorAccessorArgs<6>();

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

    uint32_t end_page = start_page + num_pages;
    for (uint32_t page_id = start_page; page_id < end_page; page_id++) {
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += wt_tiles_per_chunk) {
            const uint32_t wt_in_chunk = (wt_base + wt_tiles_per_chunk < Wt) ? wt_tiles_per_chunk : (Wt - wt_base);
            const uint32_t chunk_bytes = wt_in_chunk * tt::constants::TILE_WIDTH * elem_bytes;
            const uint32_t row_chunk_start_bytes = wt_base * tt::constants::TILE_WIDTH * elem_bytes;
            const uint32_t valid_bytes_this_chunk = (row_chunk_start_bytes >= valid_row_bytes)
                                                        ? 0
                                                        : ((valid_row_bytes - row_chunk_start_bytes) < chunk_bytes
                                                               ? (valid_row_bytes - row_chunk_start_bytes)
                                                               : chunk_bytes);

            cb_rm.reserve_back(onepage);
            if (valid_bytes_this_chunk > 0) {
                noc.async_read(
                    tensor_accessor,
                    cb_rm,
                    valid_bytes_this_chunk,
                    {.page_id = page_id, .offset_bytes = row_chunk_start_bytes},
                    {.offset_bytes = 0});
            }
            uint32_t tail_bytes_remaining = page_bytes - valid_bytes_this_chunk;
            uint32_t dst_offset_bytes = valid_bytes_this_chunk;
            while (tail_bytes_remaining > 0) {
                const uint32_t copy_bytes =
                    tail_bytes_remaining < clear_template_bytes ? tail_bytes_remaining : clear_template_bytes;
                noc.async_read(self_ep, cb_rm, copy_bytes, clear_template_src, {.offset_bytes = dst_offset_bytes});
                dst_offset_bytes += copy_bytes;
                tail_bytes_remaining -= copy_bytes;
            }
            noc.async_read_barrier();
            cb_rm.push_back(onepage);
        }
    }
}

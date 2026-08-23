// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t col_start_tile_id =
        get_arg_val<uint32_t>(1);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg_val<uint32_t>(2);
    uint32_t num_cols = get_arg_val<uint32_t>(3);  // number of cols to read

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);

    constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
    constexpr bool sfpu_two_pass = get_compile_time_arg_val(4) != 0;
    constexpr auto fp32_mode = get_compile_time_arg_val(5) != 0 ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

    constexpr uint32_t dfb_id_in0 = tt::CBIndex::c_0;

    // Two-pass statistics must process one column at a time. DEST_AUTO_LIMIT
    // interleaves multiple columns per chunk, which would feed the statistics
    // kernel tiles from the wrong columns.
    // Int32 SFPU max keeps one acc DST per column plus one shared work DST (DEST_AUTO_LIMIT - 1).
    constexpr DataFormat reduce_format = get_dataformat(dfb_id_in0);
    constexpr bool use_sfpu_reduce_path = is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format, fp32_mode>();
    constexpr uint32_t row_chunk = sfpu_two_pass ? 1
                                                 : (use_sfpu_reduce_path ? (compute_kernel_lib::DEST_AUTO_LIMIT - 1)
                                                                         : compute_kernel_lib::DEST_AUTO_LIMIT);

    constexpr uint32_t onetile = 1;
#ifdef WELFORD_TWO_PASS_STREAMING_CB_TILES
    constexpr std::uint32_t max_read_batch = WELFORD_TWO_PASS_STREAMING_CB_TILES;
#else
    constexpr std::uint32_t max_read_batch = 1;
#endif

    if constexpr (!sfpu_two_pass) {
        constexpr uint32_t dfb_id_in2 = tt::CBIndex::c_2;
        float scaler_f = __builtin_bit_cast(float, scaler_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<dfb_id_in2, REDUCE_OP, REDUCE_DIM>(scaler_f);
    }

    constexpr auto tensor_args = TensorAccessorArgs<6>();
    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

    const uint32_t tile_bytes = dfb_in0.get_tile_size();

    uint32_t w = curr_col_in_batch;
#ifndef WELFORD_TWO_PASS_L1_REPLAY
    std::uint32_t stream_write_page = 0;
#endif

    // tiles are read in the N W_skip H W_chunk order
    // W_skip(chunk size) represents the number of tile columns whose reading will be intertwined
    // H W_chunk represent tiles of the chunk read in row major order
    // exmpl. Ht = 3; Wt = 4; row_chunk = 2;
    //        read order (H, W):
    //        1. chunk:  1:(0, 0)  2:(0, 1)  3:(1, 0)   4:(1, 1)   5:(2, 0)   6:(2, 1)
    //        2. chunk:  7:(0, 2)  8:(0, 3)  9:(1, 2)  10:(1, 3)  11:(2, 2)  12:(2, 3)

    // for [N, C, W, H] tensor shape, where N != 1 or C != 1
    // chunk can contain elements with different N or C values
    // in each row we possibly need to move the col_start_tile_id to the first column of the next batch
    // reset variables are used to correctly return to the start column + repeat the process for each row
    // reset_col_start - resets col_start_tile_id to the starting column
    // reset_w - resets w to the column number in the batch of the starting column
    // reset_curr_id - resets curr_id to the next tile in the starting column
    for (uint32_t i = 0; i < num_cols; i += row_chunk) {
        uint32_t reset_curr_id = col_start_tile_id;

#ifdef WELFORD_TWO_PASS_L1_REPLAY
        for (std::uint32_t ht_base = 0; ht_base < Ht; ht_base += max_read_batch) {
            const std::uint32_t read_batch = std::min(max_read_batch, Ht - ht_base);
            dfb_in0.reserve_back(read_batch);
            for (std::uint32_t ht = 0; ht < read_batch; ++ht) {
                noc.async_read(
                    tensor_accessor,
                    dfb_in0,
                    tile_bytes,
                    {.page_id = reset_curr_id + (ht_base + ht) * Wt},
                    {.offset_bytes = ht * tile_bytes});
            }
            noc.async_read_barrier();
            dfb_in0.push_back(read_batch);
        }
        ++w;
        if (w == Wt) {
            col_start_tile_id = reset_curr_id + (Ht - 1) * Wt + 1;
            w = 0;
        } else {
            col_start_tile_id = reset_curr_id + 1;
        }
#else
        static_assert(!sfpu_two_pass || row_chunk == 1);
        if constexpr (sfpu_two_pass) {
            constexpr std::uint32_t num_passes = Ht > 4 ? 2 : 1;
            for (std::uint32_t pass = 0; pass < num_passes; ++pass) {
                const std::uint32_t pass_start = pass == 0 ? 0 : std::min(Ht, static_cast<std::uint32_t>(3));
                const std::uint32_t pass_end = pass == 0 ? Ht : Ht - 1;
                for (std::uint32_t ht_base = pass_start; ht_base < pass_end;) {
                    const std::uint32_t contiguous_pages = max_read_batch - stream_write_page;
                    const std::uint32_t read_batch =
                        std::min(std::min(max_read_batch, pass_end - ht_base), contiguous_pages);
                    dfb_in0.reserve_back(read_batch);
                    for (std::uint32_t ht = 0; ht < read_batch; ++ht) {
                        noc.async_read(
                            tensor_accessor,
                            dfb_in0,
                            tile_bytes,
                            {.page_id = reset_curr_id + (ht_base + ht) * Wt},
                            {.offset_bytes = ht * tile_bytes});
                    }
                    noc.async_read_barrier();
                    dfb_in0.push_back(read_batch);
                    ht_base += read_batch;
                    stream_write_page = (stream_write_page + read_batch) % max_read_batch;
                }
            }
            ++w;
            if (w == Wt) {
                col_start_tile_id = reset_curr_id + (Ht - 1) * Wt + 1;
                w = 0;
            } else {
                col_start_tile_id = reset_curr_id + 1;
            }
        } else {
            uint32_t chunk_end = std::min(i + row_chunk, num_cols);
            uint32_t reset_w = w;
            uint32_t reset_col_start = col_start_tile_id;
            for (uint32_t j = 0; j < Ht; ++j) {
                uint32_t curr_id = reset_curr_id + j * Wt;
                w = reset_w;
                col_start_tile_id = reset_col_start;
                for (uint32_t k = i; k < chunk_end; ++k) {
                    dfb_in0.reserve_back(onetile);
                    noc.async_read(tensor_accessor, dfb_in0, tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    dfb_in0.push_back(onetile);

                    ++w;

                    if (w == Wt) {
                        col_start_tile_id = curr_id + (Ht - j - 1) * Wt + 1;
                        curr_id = col_start_tile_id + j * Wt;
                        w = 0;
                    } else {
                        ++curr_id;
                        ++col_start_tile_id;
                    }
                }
            }
        }
#endif
    }
}

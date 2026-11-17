// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 reader (NCRISC / NoC0): per M-block, fill_halo_slab then gather_im2col_blocks.
//
// No kernel_lib helper expresses the gather: read_sticks_for_tilize reads consecutive sticks at one
// offset, while an im2col row is up to 3 taps from 3 pixels with zero-fill for out-of-image taps and
// K padding. A "gather rows by index list with zero-fill into tilize-layout pages" dataflow helper
// would close the gap. Fencing uses the local_copy helpers' transaction-id calls.
//
// Slab layout: band kh (kernel row kh's input pixels for this block) sits at slab pixel
// kh * band_stride_px. A pixel with flat index v of band kh lives at slab pixel
// v + kh * band_stride_px - (f + kh * dilation * W), f = flat index of tap (0, 0) of the block's
// first output. With band_stride_px == dilation * W the bands coincide where they overlap, so each
// input pixel is read once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t ELEM_BYTES = 2;   // bf16 slab and im2col rows
constexpr uint32_t TRID_GATHER = 4;  // bands use trids 1..3

// The NIU counts outstanding reads per transaction id in a counter capped at
// NOC_MAX_TRANSACTION_ID_COUNT; long batches are throttled to stay well below it.
constexpr uint32_t TRID_CHECK_EVERY_PACKETS = 64;
constexpr uint32_t TRID_MIN_FREE = (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2;

struct TaggedReads {
    Noc noc;
    uint32_t trid = 0;
    uint32_t packets_since_check = 0;

    FORCE_INLINE void begin(uint32_t new_trid) {
        trid = new_trid;
        packets_since_check = 0;
        dataflow_kernel_lib::set_read_trid(noc, trid);
    }

    FORCE_INLINE void read(uint64_t src_noc_addr, uint32_t dst_l1_addr, uint32_t bytes) {
        noc_async_read(src_noc_addr, dst_l1_addr, bytes);
        packets_since_check += (bytes + NOC_MAX_BURST_SIZE - 1) / NOC_MAX_BURST_SIZE;
        if (packets_since_check >= TRID_CHECK_EVERY_PACKETS) {
            packets_since_check = 0;
            while (noc_available_transactions(noc.get_noc_id(), trid) < TRID_MIN_FREE) {
            }
        }
    }

    FORCE_INLINE void wait(uint32_t which) { dataflow_kernel_lib::async_read_barrier_with_trid(noc, which); }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_halo_slab = get_compile_time_arg_val(0);
    constexpr uint32_t cb_im2col_rm = get_compile_time_arg_val(1);
    constexpr int32_t input_height = get_compile_time_arg_val(2);
    constexpr int32_t input_width = get_compile_time_arg_val(3);
    constexpr uint32_t output_width = get_compile_time_arg_val(4);
    constexpr uint32_t num_out_px = get_compile_time_arg_val(5);
    constexpr uint32_t c_in_p = get_compile_time_arg_val(6);
    constexpr int32_t padding = get_compile_time_arg_val(7);
    constexpr int32_t stride = get_compile_time_arg_val(8);
    constexpr int32_t dilation = get_compile_time_arg_val(9);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t tensor_kkh_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(13);
    constexpr int32_t band_stride_px = get_compile_time_arg_val(14);
    constexpr auto input_args = TensorAccessorArgs<15>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(3);

    constexpr uint32_t pixel_bytes = c_in_p * ELEM_BYTES;
    constexpr int32_t num_in_px = input_height * input_width;
    constexpr uint32_t block_k_cols = block_k_tiles * TILE_DIM;
    constexpr uint32_t im2col_row_bytes = block_k_cols * ELEM_BYTES;
    constexpr uint32_t tap_cols = 3 * c_in_p;
    static_assert((tensor_kkh_tiles % block_k_tiles) == 0, "a K-block never straddles a kernel row");
    static_assert(num_k_blocks * block_k_tiles == 3 * tensor_kkh_tiles, "K-blocks tile the kh blocks exactly");

    const auto input = TensorAccessor(input_args, input_addr, pixel_bytes);
    const uint32_t slab_l1 = get_write_ptr(cb_halo_slab);
    const uint64_t slab_noc = get_noc_addr(slab_l1);
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    TaggedReads reads;

    auto zero_fill = [&](uint32_t dst_l1, uint32_t bytes) {
        while (bytes > 0) {
            const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
            reads.read(zeros_noc, dst_l1, chunk);
            dst_l1 += chunk;
            bytes -= chunk;
        }
    };

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        const uint32_t block_tiles_done = m_block_idx * block_m_tiles;
        const uint32_t block_valid_tiles =
            (actual_m_tiles - block_tiles_done) < block_m_tiles ? (actual_m_tiles - block_tiles_done) : block_m_tiles;
        const uint32_t o_first = (m_tile_start + block_tiles_done) * TILE_DIM;
        const uint32_t o_end_tiles = o_first + block_valid_tiles * TILE_DIM;
        const uint32_t o_end = o_end_tiles < num_out_px ? o_end_tiles : num_out_px;

        const int32_t ho_first = o_first / output_width;
        const int32_t wo_first = o_first % output_width;
        const int32_t ho_last = (o_end - 1) / output_width;
        const int32_t wo_last = (o_end - 1) % output_width;
        const int32_t f = (ho_first * stride - padding) * input_width + wo_first * stride - padding;
        const int32_t band_len =
            (ho_last - ho_first) * stride * input_width + (wo_last - wo_first) * stride + 2 * dilation + 1;

        // ---- fill_halo_slab: band kh tagged trid kh + 1; overlapping bands read only their new tail.
        for (int32_t kh = 0; kh < 3; ++kh) {
            const int32_t seg_lo = kh == 0 ? 0
                                           : ((kh - 1) * band_stride_px + band_len > kh * band_stride_px
                                                  ? (kh - 1) * band_stride_px + band_len
                                                  : kh * band_stride_px);
            const int32_t seg_hi = kh * band_stride_px + band_len;
            const int32_t v_minus_x = f + kh * dilation * input_width - kh * band_stride_px;
            const int32_t x_lo = seg_lo > -v_minus_x ? seg_lo : -v_minus_x;
            const int32_t x_hi = seg_hi < num_in_px - v_minus_x ? seg_hi : num_in_px - v_minus_x;
            reads.begin(kh + 1);
            for (int32_t x = x_lo; x < x_hi;) {
                const uint32_t page = x + v_minus_x;
                const uint32_t end_page = x_hi + v_minus_x;
                // Interleaved runs step by the bank count, not by 1; only unit-stride runs are contiguous here.
                const uint32_t run =
                    input.contiguous_page_stride() == 1 ? input.num_contiguous_pages(page, end_page) : 1;
                reads.read(input.get_noc_addr(page), slab_l1 + x * pixel_bytes, run * pixel_bytes);
                x += run;
            }
        }

        // ---- gather_im2col_blocks: one K-block = block_m_tiles tile-rows x block_k_cols of kernel row kh.
        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const uint32_t k_tile0 = k_block_idx * block_k_tiles;
            const int32_t kh = k_tile0 / tensor_kkh_tiles;
            const uint32_t col_lo = (k_tile0 - kh * tensor_kkh_tiles) * TILE_DIM;
            const uint32_t col_hi = col_lo + block_k_cols;
            if (col_lo == 0) {
                reads.wait(kh + 1);
            }
            const uint32_t data_hi = col_hi < tap_cols ? col_hi : tap_cols;
            const int32_t kw_first = col_lo < tap_cols ? col_lo / c_in_p : 3;
            const int32_t kw_last = data_hi > col_lo ? (data_hi - 1) / c_in_p : -1;
            const int32_t slab_x_base = kh * band_stride_px - (f + kh * dilation * input_width);

            uint32_t o = o_first;
            for (uint32_t tile_row = 0; tile_row < block_m_tiles; ++tile_row) {
                cb_reserve_back(cb_im2col_rm, block_k_tiles);
                const uint32_t rows_l1 = get_write_ptr(cb_im2col_rm);
                reads.begin(TRID_GATHER);

                int32_t ho = o / output_width;
                int32_t wo = o % output_width;
                for (uint32_t r = 0; r < TILE_DIM && o < o_end; ++r, ++o) {
                    const uint32_t row_l1 = rows_l1 + r * im2col_row_bytes;
                    const int32_t h_in = ho * stride + kh * dilation - padding;
                    const int32_t w_in0 = wo * stride - padding;
                    const bool row_in_image = h_in >= 0 && h_in < input_height;
                    const int32_t slab_x0 = slab_x_base + h_in * input_width + w_in0;

                    const bool combined = dilation == 1 && row_in_image && w_in0 + kw_first >= 0 &&
                                          w_in0 + kw_last < input_width && kw_first <= kw_last;
                    if (combined) {
                        // d = 1: the touched taps are consecutive slab pixels, so one read covers them.
                        reads.read(
                            slab_noc + slab_x0 * pixel_bytes + col_lo * ELEM_BYTES,
                            row_l1,
                            (data_hi - col_lo) * ELEM_BYTES);
                    } else {
                        for (int32_t kw = kw_first; kw <= kw_last; ++kw) {
                            const uint32_t tap_lo = kw * c_in_p;
                            const uint32_t seg_lo = col_lo > tap_lo ? col_lo : tap_lo;
                            const uint32_t seg_hi = data_hi < tap_lo + c_in_p ? data_hi : tap_lo + c_in_p;
                            const uint32_t dst = row_l1 + (seg_lo - col_lo) * ELEM_BYTES;
                            const uint32_t bytes = (seg_hi - seg_lo) * ELEM_BYTES;
                            const int32_t w_in = w_in0 + kw * dilation;
                            if (row_in_image && w_in >= 0 && w_in < input_width) {
                                const int32_t slab_x = slab_x0 + kw * dilation;
                                reads.read(
                                    slab_noc + slab_x * pixel_bytes + (seg_lo - tap_lo) * ELEM_BYTES, dst, bytes);
                            } else {
                                zero_fill(dst, bytes);
                            }
                        }
                    }
                    if (col_hi > tap_cols) {
                        const uint32_t pad_lo = col_lo > tap_cols ? col_lo : tap_cols;
                        zero_fill(row_l1 + (pad_lo - col_lo) * ELEM_BYTES, (col_hi - pad_lo) * ELEM_BYTES);
                    }

                    if (++wo == static_cast<int32_t>(output_width)) {
                        wo = 0;
                        ++ho;
                    }
                }
                // Rows past the block's last valid pixel keep stale bytes: every phase downstream is
                // row-independent and the writer never stores them.
                reads.wait(TRID_GATHER);
                cb_push_back(cb_im2col_rm, block_k_tiles);
            }
        }
    }
    dataflow_kernel_lib::set_read_trid(reads.noc, 0);
}

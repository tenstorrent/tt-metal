// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

namespace {
// Broadcast reads land in a scratch slot offset by source low bits; normalize the seed value to row start before fill.
template <uint32_t element_size>
FORCE_INLINE void copy_one_element(uint32_t dst_l1_addr, uint32_t src_l1_addr) {
    static_assert(element_size == 1 || element_size == 2 || element_size == 4);

    if constexpr (element_size == 1) {
        auto* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_l1_addr);
        auto* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_l1_addr);
        dst_ptr[0] = src_ptr[0];
    } else if constexpr (element_size == 2) {
        auto* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_l1_addr);
        auto* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_l1_addr);
        dst_ptr[0] = src_ptr[0];
    } else {
        auto* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1_addr);
        auto* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1_addr);
        dst_ptr[0] = src_ptr[0];
    }
}
}  // namespace

void kernel_main() {
    uint32_t index = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);

    const uint32_t aD = get_arg_val<uint32_t>(index++);
    const uint32_t aN = get_arg_val<uint32_t>(index++);
    const uint32_t aC = get_arg_val<uint32_t>(index++);
    const uint32_t aHt = get_arg_val<uint32_t>(index++);
    const uint32_t aND = get_arg_val<uint32_t>(index++);

    const uint32_t src_addr_b = get_arg_val<uint32_t>(index++);
    const uint32_t bD = get_arg_val<uint32_t>(index++);
    const uint32_t bN = get_arg_val<uint32_t>(index++);
    const uint32_t bC = get_arg_val<uint32_t>(index++);
    const uint32_t bHt = get_arg_val<uint32_t>(index++);
    const uint32_t bND = get_arg_val<uint32_t>(index++);

    const uint32_t cHt = get_arg_val<uint32_t>(index++);
    const uint32_t cC = get_arg_val<uint32_t>(index++);
    const uint32_t cND = get_arg_val<uint32_t>(index++);
    const uint32_t current_block_start = get_arg_val<uint32_t>(index++);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(index++);
    const uint32_t row_width_elements = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_a_arg = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_b_arg = get_arg_val<uint32_t>(index++);
    const uint32_t alignment_a = get_arg_val<uint32_t>(index++);
    const uint32_t alignment_b = get_arg_val<uint32_t>(index++);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(index++);
    const uint32_t stride_size_bytes = get_arg_val<uint32_t>(index++);

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id_src);
    constexpr uint32_t element_size = src_tile_bytes / tile_hw;
    const uint32_t element_size_aligned_a = align(element_size, alignment_a);
    const uint32_t element_size_aligned_b = align(element_size, alignment_b);
    const uint32_t row_width_bytes = row_width_elements * element_size;

    const uint32_t outHt = cHt;
    const uint32_t outC = cC;
    const uint32_t outN = (aN > bN) ? aN : bN;
    const uint32_t outD = (aD > bD) ? aD : bD;
    const uint32_t outND = cND;

#if SRC_BCAST_ROW_B
    const uint32_t page_size_a = element_size_aligned_a;
    const uint32_t page_size_b = align(page_size_b_arg, alignment_b);
#else
    const uint32_t page_size_a = align(page_size_a_arg, alignment_a);
    const uint32_t page_size_b = element_size_aligned_b;
#endif

    const auto src = TensorAccessor(src_args, src_addr, page_size_a);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, page_size_b);

    const uint32_t s_h_a = (aHt == 1) ? 0 : 1;
    const uint32_t s_c_a = (aC == 1) ? 0 : aHt;
    const uint32_t s_n_a = (aN == 1) ? 0 : aC * aHt;
    const uint32_t s_d_a = (aD == 1) ? 0 : aN * aC * aHt;
    const uint32_t s_nd_a = (aND == 1) ? 0 : aD * aN * aC * aHt;

    const uint32_t s_h_b = (bHt == 1) ? 0 : 1;
    const uint32_t s_c_b = (bC == 1) ? 0 : bHt;
    const uint32_t s_n_b = (bN == 1) ? 0 : bC * bHt;
    const uint32_t s_d_b = (bD == 1) ? 0 : bN * bC * bHt;
    const uint32_t s_nd_b = (bND == 1) ? 0 : bD * bN * bC * bHt;

    const uint32_t row_blocks_per_channel = (outHt + rows_per_tile - 1) / rows_per_tile;

    uint32_t tmp = current_block_start;
    uint32_t start_th = (tmp % row_blocks_per_channel) * rows_per_tile;
    tmp /= row_blocks_per_channel;
    uint32_t start_c = tmp % outC;
    tmp /= outC;
    uint32_t start_n = tmp % outN;
    tmp /= outN;
    uint32_t start_d = tmp % outD;
    tmp /= outD;
    uint32_t start_nd = tmp;

    uint32_t row_blocks_pushed = 0;

    for (uint32_t nd = start_nd; nd < outND && row_blocks_pushed < dst_num_tiles; ++nd, start_d = 0) {
        const uint32_t ptr_nd_a = nd * s_nd_a;
        const uint32_t ptr_nd_b = nd * s_nd_b;

        for (uint32_t d = start_d; d < outD && row_blocks_pushed < dst_num_tiles; ++d, start_n = 0) {
            const uint32_t ptr_d_a = ptr_nd_a + d * s_d_a;
            const uint32_t ptr_d_b = ptr_nd_b + d * s_d_b;

            for (uint32_t n = start_n; n < outN && row_blocks_pushed < dst_num_tiles; ++n, start_c = 0) {
                const uint32_t ptr_n_a = ptr_d_a + n * s_n_a;
                const uint32_t ptr_n_b = ptr_d_b + n * s_n_b;

                for (uint32_t c = start_c; c < outC && row_blocks_pushed < dst_num_tiles; ++c, start_th = 0) {
                    const uint32_t ptr_c_a = ptr_n_a + c * s_c_a;
                    const uint32_t ptr_c_b = ptr_n_b + c * s_c_b;

                    for (uint32_t th = start_th; th < outHt && row_blocks_pushed < dst_num_tiles; th += rows_per_tile) {
                        const uint32_t rows_remaining_in_block = outHt - th;
                        const uint32_t limit =
                            (rows_remaining_in_block < rows_per_tile) ? rows_remaining_in_block : rows_per_tile;

                        const uint32_t row_block_a = ptr_c_a + th * s_h_a;
                        const uint32_t row_block_b = ptr_c_b + th * s_h_b;

                        for (uint32_t t_i = 0; t_i < tiles_per_row; ++t_i) {
                            const uint32_t current_chunk_offset = t_i * stride_size_bytes;
                            const uint32_t bytes_left_in_row = row_width_bytes - current_chunk_offset;
                            const uint32_t current_chunk_bytes =
                                (stride_size_bytes < bytes_left_in_row) ? stride_size_bytes : bytes_left_in_row;
                            const uint32_t current_chunk_elements = current_chunk_bytes / element_size;

                            cb_reserve_back(cb_id_src, 1);
                            const uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);

                            cb_reserve_back(cb_id_src_b, 1);
                            const uint32_t l1_write_addr_src_b = get_write_ptr(cb_id_src_b);

#if SRC_BCAST_ROW_B
                            const uint32_t current_read_len_b = align(current_chunk_bytes, alignment_b);

                            for (int32_t k = static_cast<int32_t>(limit) - 1; k >= 0; --k) {
                                const uint32_t row_idx_a = row_block_a + static_cast<uint32_t>(k) * s_h_a;
                                const uint64_t addr_a = get_noc_addr(row_idx_a, src);
                                const uint32_t src_low_bits = static_cast<uint32_t>(addr_a & 0xF);
                                const uint32_t scratch_l1_addr = l1_write_addr_src + src_low_bits;
                                const uint32_t row_l1_addr =
                                    l1_write_addr_src + static_cast<uint32_t>(k) * current_chunk_bytes;

                                noc_async_read(addr_a, scratch_l1_addr, element_size_aligned_a);
                                noc_async_read_barrier();

                                copy_one_element<element_size>(row_l1_addr, scratch_l1_addr);
                                FILL_TILE_WITH_FIRST_COLUMN_RM(row_l1_addr, current_chunk_elements);
                            }

                            const uint64_t addr_b = get_noc_addr(row_block_b, src_b) + current_chunk_offset;
                            noc_async_read(addr_b, l1_write_addr_src_b, current_read_len_b);
                            noc_async_read_barrier();
                            FILL_TILE_WITH_FIRST_ROW_RM(l1_write_addr_src_b, current_chunk_elements, limit);
#else
                            const uint32_t current_read_len_a = align(current_chunk_bytes, alignment_a);

                            for (int32_t k = static_cast<int32_t>(limit) - 1; k >= 0; --k) {
                                const uint32_t row_idx_b = row_block_b + static_cast<uint32_t>(k) * s_h_b;
                                const uint64_t addr_b = get_noc_addr(row_idx_b, src_b);
                                const uint32_t src_low_bits = static_cast<uint32_t>(addr_b & 0xF);
                                const uint32_t scratch_l1_addr = l1_write_addr_src_b + src_low_bits;
                                const uint32_t row_l1_addr =
                                    l1_write_addr_src_b + static_cast<uint32_t>(k) * current_chunk_bytes;

                                noc_async_read(addr_b, scratch_l1_addr, element_size_aligned_b);
                                noc_async_read_barrier();

                                copy_one_element<element_size>(row_l1_addr, scratch_l1_addr);
                                FILL_TILE_WITH_FIRST_COLUMN_RM(row_l1_addr, current_chunk_elements);
                            }

                            const uint64_t addr_a = get_noc_addr(row_block_a, src) + current_chunk_offset;
                            noc_async_read(addr_a, l1_write_addr_src, current_read_len_a);
                            noc_async_read_barrier();
                            FILL_TILE_WITH_FIRST_ROW_RM(l1_write_addr_src, current_chunk_elements, limit);
#endif

                            cb_push_back(cb_id_src, 1);
                            cb_push_back(cb_id_src_b, 1);
                        }

                        row_blocks_pushed++;
                    }
                }
            }
        }
    }
}

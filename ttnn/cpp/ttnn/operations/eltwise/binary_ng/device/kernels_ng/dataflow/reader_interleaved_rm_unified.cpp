// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

#define ALIGN_TO(len, align) (((len) + (align) - 1) / (align)) * (align)
#ifndef RM_HAS_B
#define RM_HAS_B 1
#endif
#ifndef RM_ROW_BCAST_A
#define RM_ROW_BCAST_A 0
#endif
#ifndef RM_ROW_BCAST_B
#define RM_ROW_BCAST_B 0
#endif
#ifndef RM_COL_BCAST_A
#define RM_COL_BCAST_A 0
#endif
#ifndef RM_COL_BCAST_B
#define RM_COL_BCAST_B 0
#endif
#ifndef SCALAR_OP
#define SCALAR_OP 0
#endif

void kernel_main() {
    uint32_t index = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);

    const uint32_t aD = get_arg_val<uint32_t>(index++);
    const uint32_t aN = get_arg_val<uint32_t>(index++);
    const uint32_t aC = get_arg_val<uint32_t>(index++);
    const uint32_t aHt = get_arg_val<uint32_t>(index++);
    const uint32_t aWt = get_arg_val<uint32_t>(index++);

    const uint32_t src_addr_b = get_arg_val<uint32_t>(index++);
    const uint32_t bD = get_arg_val<uint32_t>(index++);
    const uint32_t bN = get_arg_val<uint32_t>(index++);
    const uint32_t bC = get_arg_val<uint32_t>(index++);
    const uint32_t bHt = get_arg_val<uint32_t>(index++);
    const uint32_t bWt = get_arg_val<uint32_t>(index++);

    const uint32_t cHt = get_arg_val<uint32_t>(index++);
    const uint32_t cC = get_arg_val<uint32_t>(index++);
    const uint32_t current_row_start = get_arg_val<uint32_t>(index++);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(index++);
    const uint32_t row_width_elements = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_a_arg = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_b_arg = get_arg_val<uint32_t>(index++);
    const uint32_t alignment_a = get_arg_val<uint32_t>(index++);
    const uint32_t alignment_b = get_arg_val<uint32_t>(index++);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(index++);
    const uint32_t stride_size_bytes = get_arg_val<uint32_t>(index++);
    const uint32_t packed_scalar = get_arg_val<uint32_t>(index++);

    constexpr auto cb_id_src = tt::CBIndex::c_0;
#if (RM_HAS_B || SCALAR_OP)
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;
#endif
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const uint32_t tile_hw = get_tile_hw(cb_id_src);
    constexpr uint32_t element_size = src_tile_bytes / tile_hw;
    const uint32_t element_size_aligned_a = ALIGN_TO(element_size, alignment_a);
#if RM_HAS_B
    const uint32_t element_size_aligned_b = ALIGN_TO(element_size, alignment_b);
#else
    const uint32_t element_size_aligned_b = element_size_aligned_a;
#endif
    const uint32_t tile_bytes = tile_hw * element_size;
    const uint32_t row_width_bytes = row_width_elements * element_size;

    const uint32_t outHt = cHt;
    const uint32_t outC = cC;
    const uint32_t outN = (aN > bN) ? aN : bN;
    const uint32_t outD = (aD > bD) ? aD : bD;

#if RM_COL_BCAST_A
    const uint32_t page_size_a = element_size;
#else
    const uint32_t page_size_a = ALIGN_TO(page_size_a_arg, alignment_a);
#endif
#if RM_HAS_B
#if RM_COL_BCAST_B
    const uint32_t page_size_b = element_size;
#else
    const uint32_t page_size_b = ALIGN_TO(page_size_b_arg, alignment_b);
#endif
#else
    const uint32_t page_size_b = 0;
#endif

    const auto src = TensorAccessor(src_args, src_addr, page_size_a);
#if RM_HAS_B
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, page_size_b);
#endif
#if SCALAR_OP
    cb_reserve_back(cb_id_src_b, 1);
#ifdef FILL_WITH_VALUE_FLOAT_B
    const auto float_ptr_b = reinterpret_cast<const float*>(&packed_scalar);
    FILL_WITH_VALUE_FLOAT_B(cb_id_src_b, *float_ptr_b);
#endif
#ifdef FILL_WITH_VALUE_B
    FILL_WITH_VALUE_B(cb_id_src_b, packed_scalar);
#endif
    cb_push_back(cb_id_src_b, 1);
#endif

    const uint32_t s_h_a = (aHt == 1) ? 0 : 1;
    const uint32_t s_c_a = (aC == 1) ? 0 : aHt;
    const uint32_t s_n_a = (aN == 1) ? 0 : aC * aHt;
    const uint32_t s_d_a = (aD == 1) ? 0 : aN * aC * aHt;
    const uint32_t s_nd_a = (aD * aN * aC * aHt);

#if RM_HAS_B
    const uint32_t s_h_b = (bHt == 1) ? 0 : 1;
    const uint32_t s_c_b = (bC == 1) ? 0 : bHt;
    const uint32_t s_n_b = (bN == 1) ? 0 : bC * bHt;
    const uint32_t s_d_b = (bD == 1) ? 0 : bN * bC * bHt;
    const uint32_t s_nd_b = (bD * bN * bC * bHt);
#endif

    uint32_t tmp = current_row_start;
    uint32_t start_th = tmp % outHt;
    tmp /= outHt;
    uint32_t start_c = tmp % outC;
    tmp /= outC;
    uint32_t start_n = tmp % outN;
    tmp /= outN;
    uint32_t start_d = tmp % outD;
    tmp /= outD;
    uint32_t start_nd = tmp;

    uint32_t rows_in_current_tile = 0;
    bool tile_reserved = false;
    uint32_t tiles_pushed_count = 0;

    for (uint32_t nd = start_nd; nd < 1 && tiles_pushed_count < dst_num_tiles; ++nd, start_d = 0) {
        uint32_t ptr_nd_a = nd * s_nd_a;
#if RM_HAS_B
        uint32_t ptr_nd_b = nd * s_nd_b;
#endif

        for (uint32_t d = start_d; d < outD && tiles_pushed_count < dst_num_tiles; ++d, start_n = 0) {
            uint32_t ptr_d_a = ptr_nd_a + d * s_d_a;
#if RM_HAS_B
            uint32_t ptr_d_b = ptr_nd_b + d * s_d_b;
#endif

            for (uint32_t n = start_n; n < outN && tiles_pushed_count < dst_num_tiles; ++n, start_c = 0) {
                uint32_t ptr_n_a = ptr_d_a + n * s_n_a;
#if RM_HAS_B
                uint32_t ptr_n_b = ptr_d_b + n * s_n_b;
#endif

                for (uint32_t c = start_c; c < outC && tiles_pushed_count < dst_num_tiles; ++c, start_th = 0) {
                    uint32_t ptr_c_a = ptr_n_a + c * s_c_a;
#if RM_HAS_B
                    uint32_t ptr_c_b = ptr_n_b + c * s_c_b;
#endif

                    for (uint32_t th = start_th; th < outHt && tiles_pushed_count < dst_num_tiles;
                         th += rows_per_tile) {
                        uint32_t rows_remaining_in_block = outHt - th;
                        uint32_t limit =
                            (rows_remaining_in_block < rows_per_tile) ? rows_remaining_in_block : rows_per_tile;

                        uint32_t row_block_a = ptr_c_a + th * s_h_a;
#if RM_HAS_B
                        uint32_t row_block_b = ptr_c_b + th * s_h_b;
#endif

                        uint32_t r_offset = 0;
                        while (r_offset < limit && tiles_pushed_count < dst_num_tiles) {
                            if (!tile_reserved) {
                                cb_reserve_back(cb_id_src, tiles_per_row);
#if RM_HAS_B
                                cb_reserve_back(cb_id_src_b, tiles_per_row);
#endif
                                tile_reserved = true;
                                rows_in_current_tile = 0;
                            }

                            uint32_t space_in_tile = rows_per_tile - rows_in_current_tile;
                            uint32_t rows_left = limit - r_offset;
                            uint32_t chunk = (rows_left < space_in_tile) ? rows_left : space_in_tile;

                            for (uint32_t t_i = 0; t_i < tiles_per_row; ++t_i) {
                                uint32_t t_offset = t_i * stride_size_bytes;
                                uint32_t base_l1_a = get_write_ptr(cb_id_src) + (t_i * src_tile_bytes);
#if RM_HAS_B
                                uint32_t base_l1_b = get_write_ptr(cb_id_src_b) + (t_i * src_tile_bytes);
#endif

                                uint32_t curr_l1_a = base_l1_a + (rows_in_current_tile * stride_size_bytes);
#if RM_HAS_B
                                uint32_t curr_l1_b = base_l1_b + (rows_in_current_tile * stride_size_bytes);
#endif

                                uint32_t bytes_left_in_row = row_width_bytes - t_offset;
                                if (bytes_left_in_row > row_width_bytes) {
                                    bytes_left_in_row = 0;
                                }
                                uint32_t current_chunk_bytes =
                                    (stride_size_bytes < bytes_left_in_row) ? stride_size_bytes : bytes_left_in_row;
                                uint32_t current_chunk_elements = current_chunk_bytes / element_size;

#if RM_COL_BCAST_A
                                uint32_t read_len_a = element_size_aligned_a;
#else
                                uint32_t read_len_a = ALIGN_TO(current_chunk_bytes, alignment_a);
#endif
#if RM_HAS_B
#if RM_COL_BCAST_B
                                uint32_t read_len_b = element_size_aligned_b;
#else
                                uint32_t read_len_b = ALIGN_TO(current_chunk_bytes, alignment_b);
#endif
#endif

                                uint32_t chunk_l1_a = curr_l1_a;
#if RM_HAS_B
                                uint32_t chunk_l1_b = curr_l1_b;
#endif

#if RM_ROW_BCAST_A
#if RM_COL_BCAST_A
                                uint64_t addr_a = get_noc_addr(row_block_a, src);
#else
                                uint64_t addr_a = get_noc_addr(row_block_a, src) + t_offset;
#endif
                                noc_async_read(addr_a, chunk_l1_a, read_len_a);
#else
                                for (uint32_t k = 0; k < chunk; ++k) {
                                    uint32_t logical_r = r_offset + k;
                                    uint32_t row_idx_a = row_block_a + logical_r * s_h_a;
#if RM_COL_BCAST_A
                                    uint64_t addr_a = get_noc_addr(row_idx_a, src);
#else
                                    uint64_t addr_a = get_noc_addr(row_idx_a, src) + t_offset;
#endif
                                    noc_async_read(addr_a, curr_l1_a, read_len_a);
                                    curr_l1_a += current_chunk_bytes;
                                }
#endif

#if RM_HAS_B
#if RM_ROW_BCAST_B
#if RM_COL_BCAST_B
                                uint64_t addr_b = get_noc_addr(row_block_b, src_b);
#else
                                uint64_t addr_b = get_noc_addr(row_block_b, src_b) + t_offset;
#endif
                                noc_async_read(addr_b, chunk_l1_b, read_len_b);
#else
                                for (uint32_t k = 0; k < chunk; ++k) {
                                    uint32_t logical_r = r_offset + k;
                                    uint32_t row_idx_b = row_block_b + logical_r * s_h_b;
#if RM_COL_BCAST_B
                                    uint64_t addr_b = get_noc_addr(row_idx_b, src_b);
#else
                                    uint64_t addr_b = get_noc_addr(row_idx_b, src_b) + t_offset;
#endif
                                    noc_async_read(addr_b, curr_l1_b, read_len_b);
                                    curr_l1_b += current_chunk_bytes;
                                }
#endif
#endif

                                noc_async_read_barrier();
#if RM_HAS_B
#endif

#if RM_COL_BCAST_A
#if RM_ROW_BCAST_A
                                FILL_TILE_WITH_FIRST_COLUMN_RM(chunk_l1_a, current_chunk_elements);
#else
                                uint32_t fill_ptr_a = chunk_l1_a;
                                for (uint32_t k = 0; k < chunk; ++k) {
                                    FILL_TILE_WITH_FIRST_COLUMN_RM(fill_ptr_a, current_chunk_elements);
                                    fill_ptr_a += current_chunk_bytes;
                                }
#endif
#endif
#if RM_HAS_B
#if RM_COL_BCAST_B
#if RM_ROW_BCAST_B
                                FILL_TILE_WITH_FIRST_COLUMN_RM(chunk_l1_b, current_chunk_elements);
#else
                                uint32_t fill_ptr_b = chunk_l1_b;
                                for (uint32_t k = 0; k < chunk; ++k) {
                                    FILL_TILE_WITH_FIRST_COLUMN_RM(fill_ptr_b, current_chunk_elements);
                                    fill_ptr_b += current_chunk_bytes;
                                }
#endif
#endif
#endif

#if RM_ROW_BCAST_A
                                FILL_TILE_WITH_FIRST_ROW_RM(chunk_l1_a, current_chunk_elements, chunk);
#endif
#if RM_HAS_B
#if RM_ROW_BCAST_B
                                FILL_TILE_WITH_FIRST_ROW_RM(chunk_l1_b, current_chunk_elements, chunk);
#endif
#endif
#if RM_HAS_B
#endif
                            }

                            rows_in_current_tile += chunk;
                            r_offset += chunk;

                            if (rows_in_current_tile == rows_per_tile) {
                                cb_push_back(cb_id_src, tiles_per_row);
#if RM_HAS_B
                                cb_push_back(cb_id_src_b, tiles_per_row);
#endif
                                tiles_pushed_count++;
                                tile_reserved = false;
                                rows_in_current_tile = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    if (tile_reserved) {
        cb_push_back(cb_id_src, tiles_per_row);
#if RM_HAS_B
        cb_push_back(cb_id_src_b, tiles_per_row);
#endif
        tiles_pushed_count++;
        tile_reserved = false;
    }
}

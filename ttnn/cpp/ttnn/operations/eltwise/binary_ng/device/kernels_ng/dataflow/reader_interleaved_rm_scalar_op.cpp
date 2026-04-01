// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    uint32_t index = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);

    const uint32_t aD = get_arg_val<uint32_t>(index++);
    const uint32_t aN = get_arg_val<uint32_t>(index++);
    const uint32_t aC = get_arg_val<uint32_t>(index++);
    const uint32_t aHt = get_arg_val<uint32_t>(index++);
    const uint32_t aND = get_arg_val<uint32_t>(index++);

    index += 6;  // Scalar-op reader shares the RM reader ABI but does not consume tensor-B shape/address args.

    const uint32_t cHt = get_arg_val<uint32_t>(index++);
    const uint32_t cC = get_arg_val<uint32_t>(index++);
    const uint32_t cND = get_arg_val<uint32_t>(index++);
    const uint32_t current_block_start = get_arg_val<uint32_t>(index++);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(index++);
    const uint32_t row_width_elements = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_a_arg = get_arg_val<uint32_t>(index++);
    index++;  // Skip tensor-B page size.
    const uint32_t alignment_a = get_arg_val<uint32_t>(index++);
    index++;  // Skip tensor-B alignment.
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(index++);
    const uint32_t stride_size_bytes = get_arg_val<uint32_t>(index++);
    const uint32_t packed_scalar = get_arg_val<uint32_t>(index++);

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id_src);
    constexpr uint32_t element_size = src_tile_bytes / tile_hw;
    const uint32_t row_width_bytes = row_width_elements * element_size;

    const uint32_t outHt = cHt;
    const uint32_t outC = cC;
    const uint32_t outN = aN;
    const uint32_t outD = aD;
    const uint32_t outND = cND;

    const uint32_t page_size_a = align(page_size_a_arg, alignment_a);
    const auto src = TensorAccessor(src_args, src_addr, page_size_a);

    cb_reserve_back(cb_id_src_b, 1);
#ifdef FILL_WITH_VALUE_FLOAT_B
    const auto float_ptr_b = reinterpret_cast<const float*>(&packed_scalar);
    FILL_WITH_VALUE_FLOAT_B(cb_id_src_b, *float_ptr_b);
#endif
#ifdef FILL_WITH_VALUE_B
    FILL_WITH_VALUE_B(cb_id_src_b, packed_scalar);
#endif
    cb_push_back(cb_id_src_b, 1);

    const uint32_t s_h_a = (aHt == 1) ? 0 : 1;
    const uint32_t s_c_a = (aC == 1) ? 0 : aHt;
    const uint32_t s_n_a = (aN == 1) ? 0 : aC * aHt;
    const uint32_t s_d_a = (aD == 1) ? 0 : aN * aC * aHt;
    const uint32_t s_nd_a = (aND == 1) ? 0 : aD * aN * aC * aHt;

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

        for (uint32_t d = start_d; d < outD && row_blocks_pushed < dst_num_tiles; ++d, start_n = 0) {
            const uint32_t ptr_d_a = ptr_nd_a + d * s_d_a;

            for (uint32_t n = start_n; n < outN && row_blocks_pushed < dst_num_tiles; ++n, start_c = 0) {
                const uint32_t ptr_n_a = ptr_d_a + n * s_n_a;

                for (uint32_t c = start_c; c < outC && row_blocks_pushed < dst_num_tiles; ++c, start_th = 0) {
                    const uint32_t ptr_c_a = ptr_n_a + c * s_c_a;

                    for (uint32_t th = start_th; th < outHt && row_blocks_pushed < dst_num_tiles; th += rows_per_tile) {
                        const uint32_t rows_remaining_in_block = outHt - th;
                        const uint32_t limit =
                            (rows_remaining_in_block < rows_per_tile) ? rows_remaining_in_block : rows_per_tile;

                        const uint32_t row_block_a = ptr_c_a + th * s_h_a;

                        for (uint32_t t_i = 0; t_i < tiles_per_row; ++t_i) {
                            const uint32_t current_chunk_offset = t_i * stride_size_bytes;
                            const uint32_t bytes_left_in_row = row_width_bytes - current_chunk_offset;
                            const uint32_t current_chunk_bytes =
                                (stride_size_bytes < bytes_left_in_row) ? stride_size_bytes : bytes_left_in_row;
                            const uint32_t current_read_len = align(current_chunk_bytes, alignment_a);

                            cb_reserve_back(cb_id_src, 1);
                            const uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);

                            uint32_t curr_l1_a = l1_write_addr_src;
                            for (uint32_t k = 0; k < limit; ++k) {
                                const uint32_t row_idx_a = row_block_a + k * s_h_a;
                                const uint64_t addr_a = get_noc_addr(row_idx_a, src) + current_chunk_offset;
                                noc_async_read(addr_a, curr_l1_a, current_read_len);
                                curr_l1_a += current_chunk_bytes;
                            }
                            noc_async_read_barrier();

                            cb_push_back(cb_id_src, 1);
                        }

                        row_blocks_pushed++;
                    }
                }
            }
        }
    }
}

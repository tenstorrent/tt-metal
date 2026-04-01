// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t index = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(index++);
    const uint32_t row_width_elements = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);

    const uint32_t outD = get_arg_val<uint32_t>(index++);
    const uint32_t outN = get_arg_val<uint32_t>(index++);
    const uint32_t outC = get_arg_val<uint32_t>(index++);
    const uint32_t outHt = get_arg_val<uint32_t>(index++);
    const uint32_t outND = get_arg_val<uint32_t>(index++);

    const uint32_t current_block_start = get_arg_val<uint32_t>(index++);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_arg = get_arg_val<uint32_t>(index++);
    const uint32_t alignment = get_arg_val<uint32_t>(index++);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(index++);
    const uint32_t stride_size_bytes = get_arg_val<uint32_t>(index++);

    constexpr auto cb_id_out = tt::CBIndex::c_2;
    constexpr auto dst_args = TensorAccessorArgs<0>();

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id_out);
    constexpr uint32_t element_size = tile_bytes / tile_hw;
    const uint32_t full_page_size = align(page_size_arg, alignment);
    const uint32_t row_width_bytes = row_width_elements * element_size;

    const auto dst = TensorAccessor(dst_args, dst_addr, full_page_size);

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

    uint32_t row_blocks_written = 0;

    for (uint32_t nd = start_nd; nd < outND && row_blocks_written < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < outD && row_blocks_written < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < outN && row_blocks_written < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < outC && row_blocks_written < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < outHt && row_blocks_written < dst_num_tiles;
                         th += rows_per_tile) {
                        const uint32_t rows_remaining = outHt - th;
                        const uint32_t limit = (rows_remaining < rows_per_tile) ? rows_remaining : rows_per_tile;
                        const uint32_t row_block_base_row = (((((nd * outD) + d) * outN + n) * outC + c) * outHt) + th;

                        for (uint32_t t_i = 0; t_i < tiles_per_row; ++t_i) {
                            const uint32_t current_chunk_offset = t_i * stride_size_bytes;
                            const uint32_t bytes_left_in_row = row_width_bytes - current_chunk_offset;
                            const uint32_t current_chunk_bytes =
                                (stride_size_bytes < bytes_left_in_row) ? stride_size_bytes : bytes_left_in_row;
                            const uint32_t current_write_len = align(current_chunk_bytes, alignment);

                            cb_wait_front(cb_id_out, 1);

                            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                            for (uint32_t row = 0; row < limit; ++row) {
                                const uint32_t row_abs_idx = row_block_base_row + row;
                                const uint64_t dst_noc_addr = get_noc_addr(row_abs_idx, dst) + current_chunk_offset;
                                noc_async_write(l1_read_addr, dst_noc_addr, current_write_len);
                                l1_read_addr += current_chunk_bytes;
                            }

                            noc_async_write_barrier();
                            cb_pop_front(cb_id_out, 1);
                        }

                        row_blocks_written++;
                    }
                }
            }
        }
    }
}

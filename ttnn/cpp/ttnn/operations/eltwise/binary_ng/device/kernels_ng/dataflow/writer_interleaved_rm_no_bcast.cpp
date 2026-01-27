// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

#define ALIGN_TO(len, align) (((len) + (align) - 1) / (align)) * (align)
#ifndef SCALAR_OP
#define SCALAR_OP 0
#endif

void kernel_main() {
    uint32_t index = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(index++);
    const uint32_t row_width_elements = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);
    const uint32_t packed_scalar = get_arg_val<uint32_t>(index++);

    const uint32_t outD = get_arg_val<uint32_t>(index++);
    const uint32_t outN = get_arg_val<uint32_t>(index++);
    const uint32_t outC = get_arg_val<uint32_t>(index++);
    const uint32_t outHt = get_arg_val<uint32_t>(index++);
    const uint32_t outWt = get_arg_val<uint32_t>(index++);
    const uint32_t outND = get_arg_val<uint32_t>(index++);

    const uint32_t current_row_start = get_arg_val<uint32_t>(index++);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(index++);
    const uint32_t page_size_arg = get_arg_val<uint32_t>(index++);
    const uint32_t alignment = get_arg_val<uint32_t>(index++);

    constexpr auto cb_id_out = tt::CBIndex::c_2;
    constexpr auto dst_args = TensorAccessorArgs<0>();

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);
    const uint32_t tile_hw = get_tile_hw(cb_id_out);
    constexpr uint32_t element_size = tile_bytes / tile_hw;
    const uint32_t full_page_size = ALIGN_TO(page_size_arg, alignment);
    const uint32_t row_width_bytes = row_width_elements * element_size;

    const auto dst = TensorAccessor(dst_args, dst_addr, full_page_size);

    const uint32_t tiles_per_row = (row_width_elements + tile_hw - 1) / tile_hw;
    uint32_t stride_size_bytes = (row_width_bytes > tile_bytes) ? tile_bytes : ALIGN_TO(row_width_bytes, alignment);

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

    uint32_t rows_available_in_tile = 0;
    uint32_t rows_consumed_from_current_tile = 0;
    uint32_t tiles_popped_count = 0;

    uint32_t current_global_row_idx = current_row_start;

#if SCALAR_OP
    constexpr auto cb_id_scalar = tt::CBIndex::c_1;
    constexpr uint32_t onetile = 1;
    cb_reserve_back(cb_id_scalar, onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    const auto float_ptr = reinterpret_cast<const float*>(&packed_scalar);
    FILL_WITH_VALUE_FLOAT(cb_id_scalar, *float_ptr);
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_scalar, packed_scalar);
#endif
    cb_push_back(cb_id_scalar, onetile);
#endif

    for (uint32_t nd = start_nd; nd < outND && tiles_popped_count < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < outD && tiles_popped_count < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < outN && tiles_popped_count < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < outC && tiles_popped_count < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < outHt && tiles_popped_count < dst_num_tiles;
                         th += rows_per_tile) {
                        uint32_t rows_remaining = outHt - th;
                        uint32_t limit = (rows_remaining < rows_per_tile) ? rows_remaining : rows_per_tile;

                        uint32_t r_offset = 0;
                        while (r_offset < limit) {
                            if (rows_available_in_tile == 0) {
                                if (tiles_popped_count >= dst_num_tiles) {
                                    break;
                                }

                                cb_wait_front(cb_id_out, tiles_per_row);
                                rows_available_in_tile = rows_per_tile;
                                rows_consumed_from_current_tile = 0;
                            }

                            uint32_t k = (limit - r_offset);
                            if (k > rows_available_in_tile) {
                                k = rows_available_in_tile;
                            }

                            for (uint32_t t_i = 0; t_i < tiles_per_row; ++t_i) {
                                uint32_t tile_base_addr = get_read_ptr(cb_id_out) + (t_i * tile_bytes);
                                uint32_t l1_read_addr =
                                    tile_base_addr + (rows_consumed_from_current_tile * stride_size_bytes);

                                uint32_t bytes_left_in_row = row_width_bytes - (t_i * stride_size_bytes);
                                if (bytes_left_in_row > row_width_bytes) {
                                    bytes_left_in_row = 0;
                                }
                                uint32_t current_chunk_bytes =
                                    (stride_size_bytes < bytes_left_in_row) ? stride_size_bytes : bytes_left_in_row;
                                uint32_t chunk_noc_offset = t_i * stride_size_bytes;
                                uint32_t write_len = ALIGN_TO(current_chunk_bytes, alignment);

                                for (uint32_t row = 0; row < k; ++row) {
                                    uint32_t row_abs_idx = current_global_row_idx + r_offset + row;
                                    uint64_t dst_noc_addr = get_noc_addr(row_abs_idx, dst) + chunk_noc_offset;
                                    noc_async_write(l1_read_addr, dst_noc_addr, write_len);
                                    l1_read_addr += current_chunk_bytes;
                                }
                            }

                            noc_async_write_barrier();

                            r_offset += k;
                            rows_available_in_tile -= k;
                            rows_consumed_from_current_tile += k;

                            if (rows_available_in_tile == 0) {
                                cb_pop_front(cb_id_out, tiles_per_row);
                                tiles_popped_count++;
                            }
                        }

                        current_global_row_idx += limit;
                    }
                }
            }
        }
    }

    if (rows_available_in_tile > 0) {
        cb_pop_front(cb_id_out, tiles_per_row);
        tiles_popped_count++;
    }
}

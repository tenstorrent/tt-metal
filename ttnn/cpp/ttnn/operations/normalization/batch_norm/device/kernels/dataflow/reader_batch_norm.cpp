// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const auto eps = get_arg_val<uint32_t>(0);
    uint32_t src_addr = get_arg_val<uint32_t>(1);  // input tensor
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);
    uint32_t n_stride = get_arg_val<uint32_t>(5);
    uint32_t c_stride = get_arg_val<uint32_t>(6);
    uint32_t N = get_arg_val<uint32_t>(7);
    uint32_t C = get_arg_val<uint32_t>(8);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr auto cb_id_src = get_compile_time_arg_val(1);
    constexpr uint32_t onetile = 1;

    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);
    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    constexpr auto cb_id_eps = get_compile_time_arg_val(2);

    union {
        float f;
        uint32_t u;
    } scalar;
    scalar.u = eps;
    cb_reserve_back(cb_id_eps, onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    FILL_WITH_VALUE_FLOAT(cb_id_eps, scalar.f);
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);
#endif
    cb_push_back(cb_id_eps, onetile);

    // Input tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;

    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            for (uint32_t t = start_t; t < HtWt && num_tiles_read < num_tiles; ++t, ++num_tiles_read, ++tile_offset) {
                cb_reserve_back(cb_id_src, onetile);
                uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
                noc_async_read_tile(tile_offset, src, l1_write_addr_src);
                noc_async_read_barrier();
                cb_push_back(cb_id_src, onetile);
            }
            tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}

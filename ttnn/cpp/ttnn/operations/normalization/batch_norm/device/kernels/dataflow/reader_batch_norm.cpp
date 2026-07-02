// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
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

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto cb_id_eps = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr bool fill_eps_fp32 = get_compile_time_arg_val(src_args.next_compile_time_args_offset()) == 1;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t k_tile_face_elems = 1024;

    Noc noc;
    CircularBuffer cb_src(cb_id_src);
    CircularBuffer cb_eps(cb_id_eps);

    const uint32_t src_tile_bytes = cb_src.get_tile_size();
    const auto src = TensorAccessor(src_args, src_addr);

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    cb_eps.reserve_back(onetile);
    if constexpr (fill_eps_fp32) {
        float eps_f = 0;
        std::memcpy(&eps_f, &eps, sizeof(float));  // Alternative for std::bit_cast
        fill_with_val<k_tile_face_elems, float>(cb_eps.get_write_ptr(), eps_f);
    } else {
        fill_with_val_bfloat16(cb_eps.get_write_ptr(), eps);
    }
    cb_eps.push_back(onetile);

    // Input tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;

    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            for (uint32_t t = start_t; t < HtWt && num_tiles_read < num_tiles; ++t, ++num_tiles_read, ++tile_offset) {
                cb_src.reserve_back(onetile);
                noc.async_read(src, cb_src, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_src.push_back(onetile);
            }
            tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}

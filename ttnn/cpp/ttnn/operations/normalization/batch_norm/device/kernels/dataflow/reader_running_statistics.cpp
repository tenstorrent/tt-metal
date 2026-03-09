// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const auto momentum = get_arg_val<uint32_t>(0);
    uint32_t src_addr = get_arg_val<uint32_t>(1);  // input tensor
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);
    uint32_t n_stride = get_arg_val<uint32_t>(5);
    uint32_t c_stride = get_arg_val<uint32_t>(6);
    uint32_t N = get_arg_val<uint32_t>(7);
    uint32_t C = get_arg_val<uint32_t>(8);

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto cb_id_momentum = get_compile_time_arg_val(1);
    constexpr auto cb_id_one = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr uint32_t onetile = 1;

    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const auto src = TensorAccessor(src_args, src_addr, src_tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_id_src_obj(cb_id_src);
    experimental::CircularBuffer cb_id_momentum_obj(cb_id_momentum);

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;

    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t one_u = 0;
    const float one_f = 1.0f;
    std::memcpy(&one_u, &one_f, sizeof(uint32_t));  // Alternative for std::bit_cast
    fill_cb_with_value(cb_id_one, one_u);

    // momentum
    float momentum_f = 0;
    std::memcpy(&momentum_f, &momentum, sizeof(float));  // Alternative for std::bit_cast
    cb_id_momentum_obj.reserve_back(onetile);
#ifdef FILL_MOMENTUM_FP32
    fill_with_val<1024, float>(cb_id_momentum, momentum_f);
#else
    fill_with_val_bfloat16(cb_id_momentum, momentum);
#endif
    cb_id_momentum_obj.push_back(onetile);

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            for (uint32_t t = start_t; t < HtWt && num_tiles_read < num_tiles; ++t, ++num_tiles_read, ++tile_offset) {
                cb_id_src_obj.reserve_back(onetile);
                noc.async_read(src, cb_id_src_obj, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_id_src_obj.push_back(onetile);
            }
            tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}

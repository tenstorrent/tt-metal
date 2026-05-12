// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

/**
 * Reader kernel for register-based argmax over a non-HW dim (NC-style).
 *
 * For each output tile this kernel produces on its assigned core, it pushes
 * N = num_reduce_tiles data tiles into cb_in0 (input values) and N index
 * tiles into cb_in1 (each tile k entirely filled with the fp32 value (float)k).
 * The compute kernel consumes one pair at a time and tracks argmax in fp32
 * DST slots, performing a final Float32 -> UInt32 typecast before packing.
 */

namespace {

inline uint32_t compute_read_tile_id(
    uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size, bool dim_is_zero) {
    if (dim_is_zero) {
        return output_tile_id;
    }
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

// Fill a single fp32 tile in the CB at `cb_id` with the value `fill_val`
// (every element of the tile). Pushes the tile when done.
inline void push_fp32_fill_tile(uint32_t cb_id, float fill_val) {
    cb_reserve_back(cb_id, 1);
    const uint32_t write_addr = get_write_ptr(cb_id);
    const uint32_t tile_bytes = get_tile_size(cb_id);
    // fp32 is 4 bytes per element.
    const uint32_t num_elems = tile_bytes >> 2;
    volatile tt_l1_ptr float* ptr = reinterpret_cast<volatile tt_l1_ptr float*>(write_addr);
    for (uint32_t i = 0; i < num_elems; ++i) {
        ptr[i] = fill_val;
    }
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_reduce_tiles = get_arg_val<uint32_t>(3);
    const uint32_t reduce_tile_size = get_arg_val<uint32_t>(4);
    const uint32_t inner_tile_size = get_arg_val<uint32_t>(5);
    const uint32_t dim_is_zero = get_arg_val<uint32_t>(6);
    const bool dim_is_zero_b = dim_is_zero != 0;

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t onetile = 1;

    constexpr auto input_tensor_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(input_tensor_args, input_addr);

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        const uint32_t output_tile_id = start_id + out_i;
        uint32_t read_tile_id = compute_read_tile_id(output_tile_id, reduce_tile_size, inner_tile_size, dim_is_zero_b);

        for (uint32_t k = 0; k < num_reduce_tiles; ++k) {
            cb_reserve_back(cb_in0, onetile);
            const uint32_t write_ptr = get_write_ptr(cb_in0);
            noc_async_read_tile(read_tile_id, s0, write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, onetile);

            push_fp32_fill_tile(cb_in1, static_cast<float>(k));

            read_tile_id += inner_tile_size;
        }
    }
}

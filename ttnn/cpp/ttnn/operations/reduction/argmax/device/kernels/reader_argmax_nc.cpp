// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

/**
 * Reader kernel for register-based argmax over a non-HW dim (NC-style).
 *
 * For each output tile this kernel produces on its assigned core, it pushes
 * N = num_reduce_tiles value tiles into cb_in0. Indices are not staged through
 * a CB — the compute kernel materializes them as uint32 scalars inside DST via
 * fill_tile_int<UInt32>.
 */

namespace {

inline uint32_t compute_read_tile_id(
    uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size, bool dim_is_zero) {
    if (dim_is_zero) {
        return output_tile_id;
    }
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
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
    constexpr uint32_t onetile = 1;

    constexpr auto input_tensor_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(input_tensor_args, input_addr);

    Noc noc;
    CircularBuffer cb(cb_in0);
    const uint32_t tile_bytes = get_tile_size(cb_in0);

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        const uint32_t output_tile_id = start_id + out_i;
        uint32_t read_tile_id = compute_read_tile_id(output_tile_id, reduce_tile_size, inner_tile_size, dim_is_zero_b);

        for (uint32_t k = 0; k < num_reduce_tiles; ++k) {
            cb.reserve_back(onetile);
            noc.async_read(s0, cb, tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(onetile);

            read_tile_id += inner_tile_size;
        }
    }
}

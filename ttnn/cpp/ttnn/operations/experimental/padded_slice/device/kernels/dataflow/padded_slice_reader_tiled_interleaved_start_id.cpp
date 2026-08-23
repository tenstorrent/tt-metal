// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

constexpr uint32_t MAX_RANK = 8;

template <uint32_t num_tiles_per_row>
TT_KERNEL void reader(
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_tiles_per_core,
    uint32_t num_tiles_per_barrier,
    uint32_t num_tiles_per_row_this_core) {
    uint32_t num_unpadded_sticks[MAX_RANK];
    uint32_t num_padded_sticks[MAX_RANK];
    uint32_t id_per_dim[MAX_RANK];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_sticks[j] = get_vararg(j);
        num_padded_sticks[j] = get_vararg(num_dims + j);
        id_per_dim[j] = get_vararg(2 * num_dims + j);
    }

    const auto input = TensorAccessor(tensor::input);
    DataflowBuffer input_dfb(dfb::input);
    Noc noc;

    uint32_t src_stick_id = start_id;
    uint32_t tiles_read = 0;
    constexpr uint32_t tile_size = get_tile_size(dfb::input);
    const uint32_t extra_tiles_per_row = num_tiles_per_row - num_tiles_per_row_this_core;

    while (tiles_read < num_tiles_per_core) {
        input_dfb.reserve_back(num_tiles_per_barrier);
        uint32_t l1_offset = 0;
        for (uint32_t i = 0; i < num_tiles_per_barrier and tiles_read < num_tiles_per_core; ++i) {
            tiles_read++;
            if (id_per_dim[0] >= (num_unpadded_sticks[0] - extra_tiles_per_row)) {
                l1_offset += tile_size;
                src_stick_id++;
            } else {
                noc.async_read(input, input_dfb, tile_size, {.page_id = src_stick_id}, {.offset_bytes = l1_offset});
                l1_offset += tile_size;
                src_stick_id++;
            }

            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc.async_read_barrier();
        input_dfb.push_back(num_tiles_per_barrier);
    }
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t num_tiles_per_row = get_arg(args::num_tiles_per_row);

    const uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);
    const uint32_t num_tiles_per_barrier = get_arg(args::num_tiles_per_barrier);
    const uint32_t num_tiles_per_row_this_core = get_arg(args::num_tiles_per_row_this_core);

    const uint32_t num_unpadded_sticks_base = 0;
    const uint32_t num_padded_sticks_base = num_dims;
    const uint32_t id_per_dim_seed_base = 2 * num_dims;

    uint32_t src_stick_id = start_id;
    uint32_t tiles_read = 0;
    const auto s0 = TensorAccessor(tensor::src);
    const uint32_t extra_tiles_per_row = num_tiles_per_row - num_tiles_per_row_this_core;

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);
    const uint32_t tile_size = dfb_in.get_tile_size();

    // Per-dimension index counter. The host passes its starting values as varargs, which are read-only,
    // so the kernel copies them into this scratchpad and counts there.
    Scratchpad<uint32_t> id_per_dim(scratch::id_per_dim);
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(id_per_dim_seed_base + j);
    }

#ifdef DEBUG
    DPRINT(
        "num_dims: {}, start_id: {}, num_tiles_per_core: {}, num_tiles_per_barrier: {}\n",
        num_dims,
        start_id,
        num_tiles_per_core,
        num_tiles_per_barrier);

    DPRINT("tile_size: {}, src_stick_id: {}, tiles_read: {}\n", tile_size, src_stick_id, tiles_read);

    DPRINT(
        "num_unpadded_sticks: {} {} {} {}\n",
        get_vararg(num_unpadded_sticks_base + 0),
        get_vararg(num_unpadded_sticks_base + 1),
        get_vararg(num_unpadded_sticks_base + 2),
        get_vararg(num_unpadded_sticks_base + 3));
    DPRINT(
        "num_padded_sticks: {} {} {} {}\n",
        get_vararg(num_padded_sticks_base + 0),
        get_vararg(num_padded_sticks_base + 1),
        get_vararg(num_padded_sticks_base + 2),
        get_vararg(num_padded_sticks_base + 3));
    DPRINT(
        "num_tiles_per_row_this_core: {} extra_tiles_per_row: {}\n", num_tiles_per_row_this_core, extra_tiles_per_row);
#endif
    uint32_t num_tiles_pushed = 0;
    while (tiles_read < num_tiles_per_core) {
        dfb_in.reserve_back(num_tiles_per_barrier);
        uint32_t l1_offset = 0;
#ifdef DEBUG
        DPRINT("Src Buffer L1 Addr: {}\n", dfb_in.get_write_ptr());
        DPRINT("Tiles read {} Num tiles pushed: {}\n", tiles_read, num_tiles_pushed);
#endif
        for (uint32_t i = 0; i < num_tiles_per_barrier and tiles_read < num_tiles_per_core; ++i) {
            tiles_read++;
            if (id_per_dim[0] >= (get_vararg(num_unpadded_sticks_base + 0) - extra_tiles_per_row)) {
#ifdef DEBUG
                DPRINT(
                    "Skipping read for src_stick_id: {}, id_per_dim: {} {} {} {}, tiles_read: {}\n",
                    src_stick_id,
                    id_per_dim[0],
                    id_per_dim[1],
                    id_per_dim[2],
                    id_per_dim[3],
                    tiles_read);
#endif
                l1_offset += tile_size;
                src_stick_id++;

            } else {
                noc.async_read(s0, dfb_in, tile_size, {.page_id = src_stick_id}, {.offset_bytes = l1_offset});
#ifdef DEBUG
                DPRINT(
                    "src_stick_id: {}, src_buffer_l1_addr: {}, tiles_read: {}, id {} {} {} {}\n",
                    src_stick_id,
                    dfb_in.get_write_ptr() + l1_offset,
                    tiles_read,
                    id_per_dim[0],
                    id_per_dim[1],
                    id_per_dim[2],
                    id_per_dim[3]);
#endif
                l1_offset += tile_size;
                src_stick_id++;
            }

            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == get_vararg(num_unpadded_sticks_base + j)) {
                    id_per_dim[j] = 0;
                    src_stick_id += get_vararg(num_padded_sticks_base + j);
                } else {
                    break;
                }
            }
        }
        noc.async_read_barrier();
        dfb_in.push_back(num_tiles_per_barrier);
        num_tiles_pushed += num_tiles_per_barrier;
    }
}

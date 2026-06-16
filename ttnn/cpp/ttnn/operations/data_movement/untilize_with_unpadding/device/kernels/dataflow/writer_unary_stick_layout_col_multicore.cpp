
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — this kernel is op-local, used only by the untilize_with_unpadding
// multi-core COL interleaved factory). Only the access mechanism changed:
//   - the output CB id (legacy c_16) comes from the DFB consumer token (dfb::cb_id_out0)
//   - the destination address comes from the TensorAccessor binding (ta::dst)
//   - total_num_rows / ncores / third_dim / tile_width / unpadded_X_size are named compile-time args
//   - core_number / size_per_row_per_block / blocks_per_core / width_size are named runtime args
// The write_block lambda / pad logic / loop structure are preserved verbatim.
//
// NOTE: the legacy positional descriptor variant read these RTAs from indices {1,3,4,5} while the host
// only wrote indices {0..4} — i.e. it had a latent off-by-one (index 5 was an out-of-bounds read and
// index 2 was unread). The named bindings below restore the host's *intended* mapping (core_number ←
// the per-core index, size_per_row_per_block, blocks_per_core, width_size), matching the original
// pre-descriptor SetRuntimeArgs layout. See the factory's METAL2_PORT_REPORT.md for details.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = dfb::cb_id_out0;

    constexpr uint32_t total_num_rows = get_arg(args::total_num_rows);
    constexpr uint32_t ncores = get_arg(args::ncores);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    constexpr uint32_t unpadded_X_size = get_arg(args::unpadded_X_size);

    const uint32_t core_number = get_arg(args::core_number);

    const auto s = TensorAccessor(ta::dst);

    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);

    auto write_block = [&](uint32_t num_rows,
                           uint32_t mul,
                           uint32_t size_per_row_per_block,
                           uint32_t start_id,
                           uint32_t width_size,
                           uint32_t size_2d) {
        uint32_t onetile = 1;
        bool has_rows = (num_rows) > 0;

        cb_out0.wait_front(onetile * has_rows);
        uint32_t l1_read_addr = cb_out0.get_write_ptr();

        for (uint32_t k = 0; k < num_rows; k++) {
            uint32_t total_size = mul * size_per_row_per_block + start_id + width_size;
            uint32_t padded_size = total_size - unpadded_X_size;
            uint32_t write_size = width_size;

            if (mul == ncores - 1 && padded_size > 0) {
                write_size = width_size - padded_size;
            }

            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                write_size,
                {.offset_bytes = 0},
                {.page_id = size_2d + k, .offset_bytes = start_id + mul * size_per_row_per_block});

            noc.async_write_barrier();

            if (k > 0 && (k % tile_width == 0)) {
                cb_out0.pop_front(onetile * has_rows);
                cb_out0.wait_front(onetile * has_rows);
            }
            l1_read_addr += width_size;
        }

        cb_out0.pop_front(onetile * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg(args::size_per_row_per_block);
    const uint32_t blocks_per_core = get_arg(args::blocks_per_core);
    const uint32_t width_size = get_arg(args::width_size);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_id = 0;
        for (uint32_t b = 0; b < blocks_per_core; b++) {
            write_block(total_num_rows, core_number, size_per_row_per_block, start_id, width_size, size_2d);
            start_id += width_size;
        }
        size_2d += total_num_rows;
    }
}

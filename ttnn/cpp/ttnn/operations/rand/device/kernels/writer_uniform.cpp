// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"

using namespace tt;

// Metal 2.0 writer — the descriptor-era writer, ported to the spec binding namespaces and otherwise
// unchanged. Two buffers, as before: an fp32 intermediate (dfb::rand_tiles) produced by the compute
// kernel, and an output-dtype scratch (dfb::rand_out) the writer narrows into. fp32 output is written
// straight from the intermediate; bf16 output is produced by truncating each fp32 element to its high
// 16 bits (= bf16), which keeps every value strictly < `to`. The fp32-vs-bf16 path is selected by an
// OUTPUT_DTYPE_* define set host-side; only those two dtypes are supported.
//
// Only-allowed changes from the descriptor era: CB ids come from the DFB binding tokens (dfb::), the
// output address comes from the TensorAccessor binding (ta::), and start_id/num_tiles come from the
// named-arg namespace (args::) instead of positional compile-time / runtime slots. start_id/num_tiles
// are the enqueue-invariant work split; the output tensor binding is per-enqueue. The logic — the
// two-buffer structure, the OUTPUT_DTYPE_* paths, and the high-16-bits truncation — is preserved.
void kernel_main() {
    constexpr uint32_t intermed_cb_id = dfb::rand_tiles;
    constexpr uint32_t dst_cb_id = dfb::rand_out;

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t end_id = start_id + num_tiles;

    TensorAccessor out(ta::output);

    cb_reserve_back(dst_cb_id, 1);
    uint32_t dst_cb_write_ptr = get_write_ptr(dst_cb_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(intermed_cb_id, 1);

        uint32_t intermed_cb_read_ptr = get_read_ptr(intermed_cb_id);
        auto intermed_cb_addr = reinterpret_cast<float*>(intermed_cb_read_ptr);

#ifdef OUTPUT_DTYPE_FLOAT32
        noc_async_write_page(i, out, intermed_cb_read_ptr);
        noc_async_write_barrier();
        cb_pop_front(intermed_cb_id, 1);
#endif

#ifdef OUTPUT_DTYPE_BFLOAT16
        auto dst_cb_addr = reinterpret_cast<uint8_t*>(dst_cb_write_ptr);
        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                float rand_float = *intermed_cb_addr;

                uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(&rand_float) + 1;
                *(uint16_t*)dst_cb_addr = *uint16_ptr;
                dst_cb_addr += 2;
                intermed_cb_addr += 1;
            }
        }
        cb_pop_front(intermed_cb_id, 1);

        noc_async_write_page(i, out, dst_cb_write_ptr);
        noc_async_write_barrier();
#endif
    }
}

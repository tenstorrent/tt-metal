// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for the output-stationary embedding backward: for each work unit (output tile row, hidden chunk) the
// reader sends n one-hot tiles and, with each, the grad tiles of the chunk. out[t] += P x grad[t] accumulates in DEST
// (FP32 when fp32_dest_acc_en is set), and the finished chunk is packed once.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t unit_stride = get_arg_val<uint32_t>(1);
    const uint32_t total_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(2);

    constexpr uint32_t cb_grad_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_onehot_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_meta_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_out_idx = tt::CBIndex::c_16;

    CircularBuffer cb_grad(cb_grad_idx);
    CircularBuffer cb_onehot(cb_onehot_idx);
    CircularBuffer cb_meta(cb_meta_idx);
    CircularBuffer cb_out(cb_out_idx);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_onehot_idx, cb_grad_idx, cb_out_idx);
    matmul_init(cb_onehot_idx, cb_grad_idx);

    for (uint32_t u = first_unit; u < total_units; u += unit_stride) {
        constexpr uint32_t tiles = chunk_tiles;

        cb_meta.wait_front(1);
#if defined(ARCH_BLACKHOLE)
        // The reader wrote the count through L1; make sure this core does not read a stale cached copy (#11816).
        asm volatile("fence");
#endif
        const uint32_t n = read_tile_value(cb_meta_idx, 0, 0);
        cb_meta.pop_front(1);

        tile_regs_acquire();
        for (uint32_t m = 0; m < n; ++m) {
            cb_onehot.wait_front(1);
            cb_grad.wait_front(tiles);
            for (uint32_t t = 0; t < tiles; ++t) {
                matmul_tiles(cb_onehot_idx, cb_grad_idx, 0, t, t);
            }
            cb_onehot.pop_front(1);
            cb_grad.pop_front(tiles);
        }
        tile_regs_commit();

        cb_out.reserve_back(tiles);
        tile_regs_wait();
        for (uint32_t t = 0; t < tiles; ++t) {
            pack_tile(t, cb_out_idx);
        }
        tile_regs_release();
        cb_out.push_back(tiles);
    }
}

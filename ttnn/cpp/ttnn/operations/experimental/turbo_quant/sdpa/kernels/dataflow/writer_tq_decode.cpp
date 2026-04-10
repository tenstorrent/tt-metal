// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simple writer for TurboQuant SDPA decode.
// Generates identity/scale tiles using proper helpers and writes output to DRAM.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_identity_scale = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_bytes);

    // ── Generate identity/scale tile using proper helper ──
    generate_reduce_scaler(cb_identity_scale, identity_scalar_packed);

    // ── Generate column identity (same format as standard SDPA writer) ──
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // ── Write output tiles ──
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            uint32_t out_tile_id = nb * NQH * Sq_chunk_t * vDHt + nq * Sq_chunk_t * vDHt;

            cb_wait_front(cb_out, out_chunk_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < out_chunk_tiles; t++) {
                noc_async_write_tile(out_tile_id + t, out_writer, l1_read_addr);
                l1_read_addr += out_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, out_chunk_tiles);
        }
    }
}

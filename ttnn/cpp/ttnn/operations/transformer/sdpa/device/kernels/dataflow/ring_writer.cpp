// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "RING_WRITER: Starting kernel_main()" << ENDL();
    // Compile-time arguments (same as regular SDPA plus ring parameters)
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t local_q_num_chunks = get_compile_time_arg_val(7);  // Always 2 for ring distribution
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(10);
    constexpr uint32_t is_causal = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t ring_size = get_compile_time_arg_val(12);
    constexpr uint32_t ring_id = get_compile_time_arg_val(13);
    constexpr uint32_t first_chunk_id = get_compile_time_arg_val(14);
    constexpr uint32_t second_chunk_id = get_compile_time_arg_val(15);
    constexpr uint32_t global_chunk_size = get_compile_time_arg_val(16);  // In tiles
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(17);
    constexpr uint32_t scale_val = get_compile_time_arg_val(18);

    // Runtime arguments
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);

    const uint32_t local_q_chunks_per_core = local_q_end - local_q_start;  // Should be 2

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;  // Identity scale CB
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;       // Column identity CB

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, is_dram>();

    uint32_t barrier_count = 0;

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = get_dataformat(cb_out)};

    // Calculate output tensor shape
    const auto out_tile_shape = TensorTileShape(B, NQH, global_chunk_size * 2, vDHt);

    // Generate scalar values needed by compute kernel (same as regular SDPA)
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    DPRINT << "RING_WRITER: identity_scalar_packed=" << (uint32_t)identity_scalar_packed << ENDL();
    generate_bcast_col_scalar(cb_col_identity, scale_val);

    // Main processing loop: iterate over batches, heads, and local Q chunks (ring-distributed)
    DPRINT << "WRITER: is_causal=" << (uint32_t)is_causal << ENDL();
    DPRINT << "WRITER: Starting main processing loops" << ENDL();
    for (uint32_t nb = local_batch_start; nb < local_batch_end; nb++) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; nq++) {
            for (uint32_t local_q_iter = 0; local_q_iter < 2; local_q_iter++) {
                // Map local Q iteration to global Q chunk using ring balanced assignment
                uint32_t global_q_chunk = (local_q_iter == 0) ? first_chunk_id : second_chunk_id;

                DPRINT << "WRITER: Processing local_q_iter=" << local_q_iter << " global_q_chunk=" << global_q_chunk
                       << ENDL();

                // Generate causal masks for this Q chunk (same logic as regular SDPA)
                if constexpr (is_causal) {
                    uint32_t q_low_idx = global_q_chunk * Sq_chunk_t;
                    uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

                    DPRINT << "CAUSAL: q_low=" << q_low_idx << " q_high=" << q_high_idx << ENDL();

                    for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                        const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                        const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                        // Generate mask only if there's overlap (same logic as regular SDPA)
                        if (!(q_low_idx >= k_high_idx)) {
                            DPRINT << "GEN mask q=" << (uint32_t)global_q_chunk << " k=" << (uint32_t)k_chunk << ENDL();
                            generate_causal_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, global_q_chunk, k_chunk);
                        }
                    }
                }

                // Wait for compute kernel to finish processing this Q chunk
                DPRINT << "WRITER: Waiting for output from global_q_chunk=" << global_q_chunk << ENDL();
                cb_wait_front(cb_out, out_chunk_tiles);

                DPRINT << "OUTPUT READY: global_q_chunk=" << global_q_chunk << ENDL();

                // Write output tiles to global memory (same pattern as regular SDPA)
                barrier_count = 0;
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, local_q_iter * Sq_chunk_t, 0);

                for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                    for (uint32_t col = 0; col < vDHt; ++col) {
                        noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
                        ++out_tile_id;
                        l1_read_addr += tile_bytes;

                        if (++barrier_count == barrier_threshold) {
                            noc_async_writes_flushed();
                            barrier_count = 0;
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, out_chunk_tiles);

                DPRINT << "OUTPUT WRITTEN: global_q_chunk=" << global_q_chunk << ENDL();
            }
        }
    }

    noc_async_write_barrier();
    DPRINT << "WRITER COMPLETE" << ENDL();
}

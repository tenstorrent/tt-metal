// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TurboQuant SDPA decode reader kernel.
//
// Reads Q (BF16) to c_0, K/V indices (BFP4) to c_10/c_12, K/V norms to c_11/c_13.
// K indices are read NOT transposed (compute handles transpose after dequant).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t Skt = get_compile_time_arg_val(4);
    constexpr uint32_t DHt = get_compile_time_arg_val(5);
    constexpr uint32_t vDHt = get_compile_time_arg_val(6);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr bool pre_rescaled = get_compile_time_arg_val(11) == 1;

    constexpr auto q_args = TensorAccessorArgs<12>();
    constexpr auto k_idx_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_norms_args = TensorAccessorArgs<k_idx_args.next_compile_time_args_offset()>();
    constexpr auto v_idx_args = TensorAccessorArgs<k_norms_args.next_compile_time_args_offset()>();
    constexpr auto v_norms_args = TensorAccessorArgs<v_idx_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_norms_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_idx_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_norms_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    // When pre_rescaled: push KV directly to sdpa's native CBs (c_1/c_2).
    // sdpa_standard handles the data format natively (BFP8 or BFP4).
    // Reader pushes 1 chunk, sdpa consumes it, reader pushes next — pipelined.
    constexpr uint32_t cb_k_idx = pre_rescaled ? tt::CBIndex::c_1 : tt::CBIndex::c_10;
    constexpr uint32_t cb_k_norms = tt::CBIndex::c_11;
    constexpr uint32_t cb_v_idx = pre_rescaled ? tt::CBIndex::c_2 : tt::CBIndex::c_12;
    constexpr uint32_t cb_v_norms = tt::CBIndex::c_13;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_idx_tile_bytes = get_tile_size(cb_k_idx);
    constexpr uint32_t k_norms_tile_bytes = get_tile_size(cb_k_norms);
    constexpr uint32_t v_idx_tile_bytes = get_tile_size(cb_v_idx);
    constexpr uint32_t v_norms_tile_bytes = get_tile_size(cb_v_norms);

    constexpr uint32_t q_heads_per_kv = NQH / NKH;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_idx_reader = TensorAccessor(k_idx_args, k_idx_addr, k_idx_tile_bytes);
    const auto k_norms_reader = TensorAccessor(k_norms_args, k_norms_addr, k_norms_tile_bytes);
    const auto v_idx_reader = TensorAccessor(v_idx_args, v_idx_addr, v_idx_tile_bytes);
    const auto v_norms_reader = TensorAccessor(v_norms_args, v_norms_addr, v_norms_tile_bytes);

    const auto q_tile_shape = TensorTileShape(B, NQH, Sqt, DHt);
    const auto k_idx_tile_shape = TensorTileShape(B, NKH, Skt, DHt);
    const auto v_idx_tile_shape = TensorTileShape(B, NKH, Skt, vDHt);
    const auto k_norms_tile_shape = TensorTileShape(B, NKH, Skt, 1);
    const auto v_norms_tile_shape = TensorTileShape(B, NKH, Skt, 1);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // Q
            {
                const uint32_t q_start = q_tile_shape.id_of(nb, nq, 0, 0);
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_start, Sq_chunk_t, DHt, Sq_chunk_t, DHt, barrier_threshold);
            }

            const uint32_t k_head = nq / q_heads_per_kv;

            for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                const uint32_t chunk_start_row = k_chunk * Sk_chunk_t;
                const uint32_t chunk_end_row =
                    (chunk_start_row + Sk_chunk_t < Skt) ? chunk_start_row + Sk_chunk_t : Skt;
                const uint32_t kv_row_count = chunk_end_row - chunk_start_row;

                // K indices: transposed when pre_rescaled (compute skips dequant),
                // NOT transposed otherwise (compute transposes after dequant).
                {
                    const uint32_t k_start = k_idx_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<k_idx_tile_bytes>(
                        k_idx_reader,
                        cb_k_idx,
                        k_start,
                        kv_row_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        barrier_threshold,
                        pre_rescaled  // transpose K when pre_rescaled
                    );
                }

                // K norms (skip when pre_rescaled — values already include norms)
                if constexpr (!pre_rescaled) {
                    const uint32_t n_start = k_norms_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<k_norms_tile_bytes>(
                        k_norms_reader, cb_k_norms, n_start, kv_row_count, 1, Sk_chunk_t, 1, barrier_threshold);
                }

                // V indices: NOT transposed
                {
                    const uint32_t v_start = v_idx_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<v_idx_tile_bytes>(
                        v_idx_reader, cb_v_idx, v_start, kv_row_count, vDHt, Sk_chunk_t, vDHt, barrier_threshold);
                }

                // V norms (skip when pre_rescaled)
                if constexpr (!pre_rescaled) {
                    const uint32_t n_start = v_norms_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<v_norms_tile_bytes>(
                        v_norms_reader, cb_v_norms, n_start, kv_row_count, 1, Sk_chunk_t, 1, barrier_threshold);
                }
            }
        }
    }
}

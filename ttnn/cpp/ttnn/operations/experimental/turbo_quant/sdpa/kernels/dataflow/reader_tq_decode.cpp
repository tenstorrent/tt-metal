// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TurboQuant SDPA decode reader kernel — BF16 direct push version.
//
// For MVP: reads BF16 K/V tensors and pushes directly to cb_k_in/cb_v_in
// (the SDPA input CBs). Also pushes raw BFP4 indices and norms to their CBs
// for the compute kernel to discard (maintaining pipeline compatibility).
//
// Inputs:
//   Q          (BF16, interleaved)  [B, NQH, 1, DH]
//   K_bf16     (BF16, interleaved)  [B, NKH, Sk, DH]   -- direct K values
//   K norms    (BF16, interleaved)  [B, NKH, Sk, 1]    -- unused but consumed
//   V_bf16     (BF16, interleaved)  [B, NKH, Sk, vDH]  -- direct V values
//   V norms    (BF16, interleaved)  [B, NKH, Sk, 1]    -- unused but consumed

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    // ── Compile-time args ──
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

    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_idx_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_norms_args = TensorAccessorArgs<k_idx_args.next_compile_time_args_offset()>();
    constexpr auto v_idx_args = TensorAccessorArgs<k_norms_args.next_compile_time_args_offset()>();
    constexpr auto v_norms_args = TensorAccessorArgs<v_idx_args.next_compile_time_args_offset()>();

    // ── Runtime args ──
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

    // ── CB indices ──
    // Push K/V directly to SDPA input CBs (c_1, c_2)
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;  // Direct K push
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;  // Direct V push

    // ── Tile sizes ──
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    // For MVP: K/V are BF16, same tile size as Q
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    // ── Derived constants ──
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t q_heads_per_kv = NQH / NKH;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    // ── Tensor accessors ──
    // k_idx and v_idx accessors are used for the K/V BF16 tensors
    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_reader = TensorAccessor(k_idx_args, k_idx_addr, k_tile_bytes);
    const auto v_reader = TensorAccessor(v_idx_args, v_idx_addr, v_tile_bytes);

    // Tile shapes
    const auto q_tile_shape = TensorTileShape(B, NQH, Sqt, DHt);
    const auto k_tile_shape = TensorTileShape(B, NKH, Skt, DHt);
    const auto v_tile_shape = TensorTileShape(B, NKH, Skt, vDHt);

    // ── Main loop ──
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // ── Read Q ──
            {
                const uint32_t q_start = q_tile_shape.id_of(nb, nq, 0, 0);
                read_chunk_with_padding<q_tile_bytes>(
                    q_reader, cb_q_in, q_start, Sq_chunk_t, DHt, Sq_chunk_t, DHt, barrier_threshold);
            }

            const uint32_t k_head = nq / q_heads_per_kv;

            // ── Push K/V directly to SDPA CBs chunk by chunk ──
            for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                const uint32_t chunk_start_row = k_chunk * Sk_chunk_t;
                const uint32_t chunk_end_row =
                    (chunk_start_row + Sk_chunk_t < Skt) ? chunk_start_row + Sk_chunk_t : Skt;
                const uint32_t kv_row_count = chunk_end_row - chunk_start_row;

                // ── K: read BF16 directly into cb_k_in, TRANSPOSED ──
                // K must be stored transposed in CB for Q×K^T matmul (matches standard SDPA)
                {
                    const uint32_t k_start = k_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<k_tile_bytes>(
                        k_reader,
                        cb_k_in,
                        k_start,
                        kv_row_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        barrier_threshold,
                        true  // transpose=true for K reads
                    );
                }

                // ── V: read BF16 directly into cb_v_in ──
                {
                    const uint32_t v_start = v_tile_shape.id_of(nb, k_head, chunk_start_row, 0);
                    read_chunk_with_padding<v_tile_bytes>(
                        v_reader, cb_v_in, v_start, kv_row_count, vDHt, Sk_chunk_t, vDHt, barrier_threshold);
                }
            }
        }
    }
}

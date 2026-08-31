// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa writer (forked from sparse_sdpa_msa_writer): co-gather lower K/V tile halves of each chunk, write
// the output tiles, and build the persistent compute tiles (reduce scaler, col-identity, -inf mask).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (shared with sparse_sdpa_msa)
#include "dataflow_common.hpp"         // fill_neginf_tile (persistent -inf mask tile)

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16

void kernel_main() {
    constexpr uint32_t m = get_compile_time_arg_val(0);
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t out_tiles_per_work = get_compile_time_arg_val(2);
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(3);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(4);
    constexpr uint32_t k_half = get_compile_time_arg_val(5);
    constexpr uint32_t v_half = get_compile_time_arg_val(6);
    constexpr uint32_t k_head_stride = get_compile_time_arg_val(7);
    constexpr uint32_t v_head_stride = get_compile_time_arg_val(8);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(11);

    // CB ids match the factory's writer compile-arg block.
    constexpr uint32_t cb_out_im = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(13);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(14);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(15);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(16);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(17);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(18);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(19);

    constexpr auto out_args = TensorAccessorArgs<20, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t work_start = get_arg_val<uint32_t>(3);
    const uint32_t work_count = get_arg_val<uint32_t>(4);

    Noc noc;
    experimental::CB out_cb(cb_out_im), k_cb(cb_k_in), v_cb(cb_v_in), kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);

    // Reduce identity scaler; the softmax scale is applied in compute.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();

    // Col-identity for the final row-sum reduction.
    generate_bcast_col_scalar(experimental::CB(cb_col_identity), one_bf16_packed);

    // Persistent all -inf tile for the ragged-block masks.
    {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_neginf);
        experimental::CB(cb_neginf).reserve_back(1);
        fill_neginf_tile<mask_tile_bytes>(cb_neginf, 0);
        experimental::CB(cb_neginf).push_back(1);
    }

    for (uint32_t work = 0; work < work_count; ++work) {
        const uint32_t w = work_start + work;
        const uint32_t head = w / n_q_tiles;
        const uint32_t k_base = head * k_head_stride;
        const uint32_t v_base = head * v_head_stride;

        // Co-gather the lower K/V tile halves of each chunk the reader requests, then ack.
        bool last = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t n_valid;
            uint32_t block_ids[m];
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                n_valid = rq[0];
                last = rq[1] != 0;
                for (uint32_t b = 0; b < n_valid; ++b) {
                    block_ids[b] = rq[2 + b];
                }
            }
            kreq_cb.pop_front(1);

            sparse_sdpa_msa::TridRing ring{noc};  // K/V lower halves share one ring
            for (uint32_t b = 0; b < n_valid; ++b) {
                const uint32_t k_tile0 = k_base + block_ids[b] * k_tiles_per_block;
                const uint32_t v_tile0 = v_base + block_ids[b] * v_tiles_per_block;
                const uint32_t k_off = b * k_tiles_per_block;
                const uint32_t v_off = b * v_tiles_per_block;
                for (uint32_t i = 0; i < k_half; ++i) {
                    ring.read(k, k_cb, k_tile_bytes, k_tile0 + i, (k_off + i) * k_tile_bytes);
                }
                for (uint32_t i = 0; i < v_half; ++i) {
                    ring.read(v, v_cb, v_tile_bytes, v_tile0 + i, (v_off + i) * v_tile_bytes);
                }
            }
            ring.drain();
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }

        // Output stays TILE: one contiguous run of Sqt*vDHt pages per work item (head-major layout).
        out_cb.wait_front(out_tiles_per_work);
        for (uint32_t i = 0; i < out_tiles_per_work; ++i) {
            noc.async_write(
                out_cb, out, out_tile_bytes, {.offset_bytes = i * out_tile_bytes}, {.page_id = w * out_tiles_per_work + i});
        }
        noc.async_write_barrier();
        out_cb.pop_front(out_tiles_per_work);
    }
}

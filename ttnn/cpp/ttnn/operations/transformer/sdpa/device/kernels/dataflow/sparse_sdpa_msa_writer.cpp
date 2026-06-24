// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa writer: co-gather lower K/V tile halves, write row-major output, and build persistent compute
// tiles used by softmax normalization.

#include <stdint.h>
#include <tt-metalium/constants.hpp>  // tt::constants::TILE_WIDTH
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (K_TRID_RING knob)

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H_logical = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t n_kv = get_compile_time_arg_val(4);
    // CB ids match the factory's writer compile-arg block.
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(6);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(7);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(8);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(9);  // bf16
    // Writer fills the lower half of each block's K/V tiles into reader-reserved CBs.
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(10);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(11);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(12);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(13);
    constexpr uint32_t block_size = get_compile_time_arg_val(14);
    constexpr uint32_t d = get_compile_time_arg_val(15);
    constexpr uint32_t v_dim = get_compile_time_arg_val(16);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(18);
    constexpr auto out_args = TensorAccessorArgs<19, 0>();
    // K/V use RuntimeTensorShape so T can vary without recompilation.
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_start = get_arg_val<uint32_t>(1);
    const uint32_t work_count = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t k_batch_tile_offset = get_arg_val<uint32_t>(5);  // cache_batch_idx slot offset (0 if not indexed)
    const uint32_t v_batch_tile_offset = get_arg_val<uint32_t>(6);
    uint32_t k_group_tile_stride = 0;
    uint32_t v_group_tile_stride = 0;
    if constexpr (n_kv > 1) {
        k_group_tile_stride = get_arg_val<uint32_t>(7);
        v_group_tile_stride = get_arg_val<uint32_t>(8);
    }

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // v_dim bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,v_dim] block
    // K/V tile-grid geometry; writer gathers [0, half).
    constexpr uint32_t Skt = block_size / 32;
    constexpr uint32_t DHt = d / 32;
    constexpr uint32_t vDHt_k = v_dim / 32;
    constexpr uint32_t k_tiles_per_block = Skt * DHt;
    constexpr uint32_t v_tiles_per_block = Skt * vDHt_k;
    constexpr uint32_t k_half = k_tiles_per_block >> 1;
    constexpr uint32_t v_half = v_tiles_per_block >> 1;

    Noc noc;
    experimental::CB out_cb(cb_out_rm), k_cb(cb_k_in), v_cb(cb_v_in), kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);

    // Reduce identity scaler; softmax scale is applied in compute.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1,
        /*compute_uses_reduce_tile=*/true>();

    // Col-identity for final row-sum reduction.
    generate_bcast_col_scalar(cb_col_identity, one_bf16_packed);

    // Boundary-mask tile. Current block-level selection always makes this path inert.
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < 16; ++c) {
            p[c] = neg_inf_bf16;        // face0 row 0 (cols 0-15)
            p[256 + c] = neg_inf_bf16;  // face1 row 0 (cols 16-31)
        }
        neginf_cb.push_back(1);
    }

    uint32_t tok = work_start;
    uint32_t kv_group = 0;
    if constexpr (n_kv > 1) {
        kv_group = work_start / S;
        tok = work_start - kv_group * S;
    }
    for (uint32_t work = 0; work < work_count; ++work) {
        // Co-gather lower K/V tile halves for each selected block, then ack the reader.
        bool last = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t block_id;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                block_id = rq[0];
                last = rq[1] != 0;
            }
            kreq_cb.pop_front(1);
            uint32_t k_tile0 = k_batch_tile_offset + block_id * k_tiles_per_block;
            uint32_t v_tile0 = v_batch_tile_offset + block_id * v_tiles_per_block;
            if constexpr (n_kv > 1) {
                k_tile0 += kv_group * k_group_tile_stride;
                v_tile0 += kv_group * v_group_tile_stride;
            }
            sparse_sdpa_msa::TridRing ring{noc};  // K/V lower halves share one ring.
            for (uint32_t i = 0; i < k_half; ++i) {
                ring.read(k, k_cb, k_tile_bytes, k_tile0 + i, i * k_tile_bytes);
            }
            for (uint32_t i = 0; i < v_half; ++i) {
                ring.read(v, v_cb, v_tile_bytes, v_tile0 + i, i * v_tile_bytes);
            }
            ring.drain();
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }

        out_cb.wait_front(block_tiles);  // one untilized [H, v_dim] block
        for (uint32_t h = 0; h < H_logical; ++h) {
            // Row h at CB byte offset h*row_bytes maps to output head h within this KV group.
            uint32_t out_head = h;
            if constexpr (n_kv > 1) {
                out_head += kv_group * H_logical;
            }
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = out_head * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);

        ++tok;
        if constexpr (n_kv > 1) {
            if (tok == S) {
                tok = 0;
                ++kv_group;
            }
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa packed-group writer: builds the persistent compute tiles (reduce scaler, col identity, full and
// half -inf mask tiles), co-gathers the lower K/V tile halves of every union block into the slot the reader
// reserved (address in the request), and writes each group's output rows (two tokens per untilized 32-row tile
// row) to the row-major output.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (K_TRID_RING knob)
#include "dataflow_common.hpp"         // fill_neginf_tile

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16

// bf16 tile with -inf on the faces of `half` (0: faces 0,1 = rows 0-15; 1: faces 2,3 = rows 16-31), 0 elsewhere.
FORCE_INLINE void fill_half_neginf_tile_bf16(uint32_t l1_addr, uint32_t half) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t f = 0; f < 4; ++f) {
        const uint32_t val = ((f >> 1) == half) ? 0xFF80FF80u : 0u;
        for (uint32_t i = 0; i < 128; ++i) {
            ptr[f * 128 + i] = val;
        }
    }
}

void kernel_main() {
    constexpr uint32_t H_logical = get_compile_time_arg_val(0);  // 16
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t n_kv = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);  // one output row (v_dim * elem)
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);  // vDHt: tiles per untilized tile row
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(6);
    constexpr uint32_t k_half = get_compile_time_arg_val(7);
    constexpr uint32_t v_half = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(9);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(10);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(11);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(12);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(13);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(16);
    constexpr uint32_t cb_halfmask = get_compile_time_arg_val(17);
    constexpr auto out_args = TensorAccessorArgs<18, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    // Runtime args: same slots as the legacy writer (SparseSDPAMsaOperation::WriterArg). The K/V offsets are
    // folded into the physical block the reader sends, except the per-group/batch tile offsets below.
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_start = get_arg_val<uint32_t>(1);
    const uint32_t work_count = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t k_batch_tile_offset = get_arg_val<uint32_t>(5);
    const uint32_t v_batch_tile_offset = get_arg_val<uint32_t>(6);
    uint32_t k_group_tile_stride = 0;
    uint32_t v_group_tile_stride = 0;
    if constexpr (n_kv > 1) {
        k_group_tile_stride = get_arg_val<uint32_t>(7);
        v_group_tile_stride = get_arg_val<uint32_t>(8);
    }

    Noc noc;
    experimental::CB out_cb(cb_out_rm), kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);

    // Reduce identity scaler; softmax scale is applied in compute.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    // Col-identity for final row-sum reduction.
    generate_bcast_col_scalar(experimental::CB(cb_col_identity), one_bf16_packed);
    // Persistent mask tiles: all -inf (both tokens of a tile row hidden), top-half / bottom-half -inf.
    {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_neginf);
        experimental::CB(cb_neginf).reserve_back(1);
        fill_neginf_tile<mask_tile_bytes>(cb_neginf, 0);
        experimental::CB(cb_neginf).push_back(1);
        experimental::CB hm(cb_halfmask);
        hm.reserve_back(2);
        fill_half_neginf_tile_bf16(hm.get_write_ptr(), 0);
        fill_half_neginf_tile_bf16(hm.get_write_ptr() + mask_tile_bytes, 1);
        hm.push_back(2);
    }

    uint32_t tok = work_start;
    uint32_t kv_group = 0;
    if constexpr (n_kv > 1) {
        kv_group = work_start / S;
        tok = work_start - kv_group * S;
    }
    uint32_t remaining = work_count;
    while (remaining > 0) {
        // Co-gather the lower K/V halves of each union block of the group, then ack the reader.
        bool last = false;
        uint32_t g = 0;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t phys_block, k_slot, v_slot;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                phys_block = rq[0];
                last = rq[1] != 0;
                k_slot = rq[2];
                v_slot = rq[3];
                g = rq[4];
            }
            kreq_cb.pop_front(1);
            uint32_t k_tile0 = k_batch_tile_offset + phys_block * k_tiles_per_block;
            uint32_t v_tile0 = v_batch_tile_offset + phys_block * v_tiles_per_block;
            if constexpr (n_kv > 1) {
                k_tile0 += kv_group * k_group_tile_stride;
                v_tile0 += kv_group * v_group_tile_stride;
            }
            sparse_sdpa_msa::TridRing ring{noc};  // K/V lower halves share one ring.
            for (uint32_t i = 0; i < k_half; ++i) {
                ring.read_to(k, k_slot + i * k_tile_bytes, k_tile_bytes, k_tile0 + i);
            }
            for (uint32_t i = 0; i < v_half; ++i) {
                ring.read_to(v, v_slot + i * v_tile_bytes, v_tile_bytes, v_tile0 + i);
            }
            ring.drain();
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }

        // Output: tile row r holds slot 2r (rows 0-15) and slot 2r+1 (rows 16-31), each 16 heads.
        const uint32_t n_rows = (g + 1) / 2;
        for (uint32_t r = 0; r < n_rows; ++r) {
            out_cb.wait_front(row_tiles);
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t s = 2 * r + half;
                if (s >= g) {
                    break;
                }
                for (uint32_t h = 0; h < H_logical; ++h) {
                    const uint32_t out_head = h + kv_group * H_logical;
                    noc.async_write(
                        out_cb,
                        out,
                        row_bytes,
                        {.offset_bytes = (half * H_logical + h) * row_bytes},
                        {.page_id = out_head * S + tok + s});
                }
            }
            noc.async_write_barrier();
            out_cb.pop_front(row_tiles);
        }

        tok += g;
        remaining -= g;
        if constexpr (n_kv > 1) {
            if (tok == S) {
                tok = 0;
                ++kv_group;
            }
        }
    }
}

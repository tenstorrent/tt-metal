// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v3) writer.
// LEADER: serves the leader reader's kreq by fetching each block's K tiles from DRAM into the
// shared stream slots (the second NoC of the fetch pair). Nothing else.
// WORKER: builds the persistent compute tiles, loads each pass's resident Q rows, serves the local
// reader's kreq by pulling K from the leader's L1 slot, and drains normalized row outputs to DRAM.
// K-serving and output-draining are polled in one loop (blocking either side alone would deadlock).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"
#include "dataflow_common.hpp"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;

void kernel_main() {
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(0);  // S / 64 per head
    constexpr uint32_t R_MAX = get_compile_time_arg_val(1);
    constexpr uint32_t q_tiles_per_row = get_compile_time_arg_val(2);    // Sqt * DHt
    constexpr uint32_t out_tiles_per_row = get_compile_time_arg_val(3);  // Sqt * vDHt
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(4);
    constexpr uint32_t k_head_stride = get_compile_time_arg_val(5);
    constexpr uint32_t q_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(8);

    constexpr uint32_t cb_q_res = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k_stream = get_compile_time_arg_val(10);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(11);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(12);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(13);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(14);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(15);
    constexpr uint32_t cb_qdone = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out = get_compile_time_arg_val(17);

    constexpr auto out_args = TensorAccessorArgs<18, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto q_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    uint32_t argi = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t q_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t head = get_arg_val<uint32_t>(argi++);
    const uint32_t is_leader = get_arg_val<uint32_t>(argi++);
    const uint32_t n_passes = get_arg_val<uint32_t>(argi++);
    const uint32_t leader_x = get_arg_val<uint32_t>(argi++);
    const uint32_t leader_y = get_arg_val<uint32_t>(argi++);
    const uint32_t row_start = get_arg_val<uint32_t>(argi++);
    const uint32_t row_stride = get_arg_val<uint32_t>(argi++);
    const uint32_t row_count = get_arg_val<uint32_t>(argi++);

    Noc noc;
    experimental::CB q_cb(cb_q_res), k_cb(cb_k_stream), kreq_cb(cb_kreq), kack_cb(cb_kack);
    experimental::CB qdone_cb(cb_qdone), out_cb(cb_out);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto q = TensorAccessor(q_args, q_addr);

    if (is_leader) {
        // Fetch K for every kreq {block_id, slot} from DRAM into the shared stream slots;
        // a sentinel kreq ends each pass.
        const uint32_t k_base = head * k_head_stride;
        for (uint32_t pass = 0; pass < n_passes; ++pass) {
            while (true) {
                kreq_cb.wait_front(1);
                uint32_t block_id, slot;
                {
                    volatile tt_l1_ptr uint32_t* rq =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                    block_id = rq[0];
                    slot = rq[1];
                }
                kreq_cb.pop_front(1);
                if (block_id == 0xFFFFFFFFu) {
                    break;
                }
                sparse_sdpa_msa::TridRing ring{noc};
                const uint32_t k_tile0 = k_base + block_id * k_tiles_per_block;
                for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
                    ring.read(k, k_cb, k_tile_bytes, k_tile0 + i, (slot * k_tiles_per_block + i) * k_tile_bytes);
                }
                ring.drain();
                kack_cb.reserve_back(1);
                kack_cb.push_back(1);
            }
        }
        return;
    }

    // ---------------- WORKER ----------------
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(experimental::CB(cb_col_identity), one_bf16_packed);
    {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_neginf);
        experimental::CB(cb_neginf).reserve_back(1);
        fill_neginf_tile<mask_tile_bytes>(cb_neginf, 0);
        experimental::CB(cb_neginf).push_back(1);
    }

    const uint32_t k_l1_base = k_cb.get_write_ptr();

    uint32_t drained = 0;
    uint32_t pass_base = 0;

    auto serve_kreq_if_any = [&]() {
        if (!cb_pages_available_at_front(cb_kreq, 1)) {
            return;
        }
        kreq_cb.wait_front(1);
        uint32_t leader_slot, slot;
        {
            volatile tt_l1_ptr uint32_t* rq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
            leader_slot = rq[0];
            slot = rq[1];
        }
        kreq_cb.pop_front(1);
        const uint64_t src = get_noc_addr(
            leader_x, leader_y, k_l1_base + leader_slot * k_tiles_per_block * k_tile_bytes, noc.get_noc_id());
        noc_async_read(
            src, k_l1_base + slot * k_tiles_per_block * k_tile_bytes, k_tiles_per_block * k_tile_bytes,
            noc.get_noc_id());
        noc.async_read_barrier();
        kack_cb.reserve_back(1);
        kack_cb.push_back(1);
    };

    while (pass_base < row_count || drained < row_count) {
        if (pass_base < row_count && drained >= pass_base) {
            const uint32_t pass_rows = (row_count - pass_base < R_MAX) ? (row_count - pass_base) : R_MAX;
            for (uint32_t r = 0; r < pass_rows; ++r) {
                const uint32_t ri = pass_base + r;  // chunk-cyclic (see reader)
                const uint32_t q_tile = row_start + (ri >> 2) * row_stride + (ri & 3);
                const uint32_t page0 = (head * n_q_tiles + q_tile) * q_tiles_per_row;
                for (uint32_t i = 0; i < q_tiles_per_row; ++i) {
                    // cb_q_res is RAM-mode: never reserved/pushed here, offsets from the base.
                    noc.async_read(
                        q, q_cb, q_tile_bytes, {.page_id = page0 + i},
                        {.offset_bytes = (r * q_tiles_per_row + i) * q_tile_bytes});
                }
                serve_kreq_if_any();
            }
            noc.async_read_barrier();
            qdone_cb.reserve_back(1);
            qdone_cb.push_back(1);
            pass_base += pass_rows;
        }

        serve_kreq_if_any();

        if (cb_pages_available_at_front(cb_out, out_tiles_per_row)) {
            out_cb.wait_front(out_tiles_per_row);
            const uint32_t q_tile = row_start + (drained >> 2) * row_stride + (drained & 3);
            const uint32_t page0 = (head * n_q_tiles + q_tile) * out_tiles_per_row;
            for (uint32_t i = 0; i < out_tiles_per_row; ++i) {
                noc.async_write(
                    out_cb, out, out_tile_bytes, {.offset_bytes = i * out_tile_bytes}, {.page_id = page0 + i});
            }
            noc.async_write_barrier();
            out_cb.pop_front(out_tiles_per_row);
            ++drained;
        }
    }
}

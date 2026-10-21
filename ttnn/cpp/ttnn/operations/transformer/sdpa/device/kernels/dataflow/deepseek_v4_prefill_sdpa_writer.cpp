// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_sdpa writer: builds the persistent compute-input tiles (reduce scaler, col-identity,
// all-(-inf) tile and the optional per-head sinks), gathers its half [0, split) of every K chunk the reader
// requests from window_kv or cache_kv into the shared cb_k_rm, and drains each token's untilized [H, V_DIM] block
// to the ROW_MAJOR [1, H, S, V_DIM] output.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "deepseek_v4_prefill_sdpa_gather.hpp"

namespace ds = deepseek_v4_prefill_sdpa;

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(ds::writer_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(ds::writer_ct_arg::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(ds::writer_ct_arg::V_DHT);
    constexpr uint32_t k_dim = get_compile_time_arg_val(ds::writer_ct_arg::K_DIM);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(ds::writer_ct_arg::CB_OUT_RM);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(ds::writer_ct_arg::CB_SCALE);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(ds::writer_ct_arg::CB_COL_IDENTITY);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(ds::writer_ct_arg::CB_NEGINF);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(ds::writer_ct_arg::CB_IDX);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(ds::writer_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(ds::writer_ct_arg::CB_KACK);
    constexpr bool use_attention_sink = get_compile_time_arg_val(ds::writer_ct_arg::USE_ATTENTION_SINK) != 0;
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(ds::writer_ct_arg::CB_ATTENTION_SINK);
    constexpr uint32_t cb_sink_scratch = get_compile_time_arg_val(ds::writer_ct_arg::CB_SINK_SCRATCH);

    constexpr auto out_args = TensorAccessorArgs<ds::writer_ct_arg::END, ds::writer_common_arg::END>();
    constexpr auto window_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto cache_args = TensorAccessorArgs<
        window_args.next_compile_time_args_offset(),
        window_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(ds::writer_common_arg::OUTPUT_ADDRESS);
    const uint32_t window_addr = get_common_arg_val<uint32_t>(ds::writer_common_arg::WINDOW_ADDRESS);
    const uint32_t cache_addr = get_common_arg_val<uint32_t>(ds::writer_common_arg::CACHE_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t k_row_bytes = k_dim * sizeof(uint16_t);
    constexpr uint32_t out_row_bytes = vDHt * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    // Compute's blocks cover whole tiles of heads; only the first H rows are real.
    constexpr uint32_t Sqt = (H + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    constexpr uint32_t block_tiles = Sqt * vDHt;

    Noc noc;
    experimental::CB out_cb(cb_out_rm), idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto window = TensorAccessor(window_args, window_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    // The reader's key list (it owns the reservation; both kernels see the same L1).
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    // Reduce identity scaler (1.0; the softmax scale is applied in compute's exp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();

    // Col-identity (column 0 = 1.0): finalizes the within-tile row-sum in normalize_row_streaming.
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);

    // All-(-inf) tile, row 0 only (compute row-broadcasts it onto fully masked key tiles).
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = neg_inf_bf16;
            p[tt::constants::FACE_HW + c] = neg_inf_bf16;
        }
        neginf_cb.push_back(1);
    }

    if constexpr (use_attention_sink) {
        if (tok_count > 0) {
            constexpr auto sink_args = TensorAccessorArgs<
                cache_args.next_compile_time_args_offset(),
                cache_args.next_common_runtime_args_offset()>();
            const auto sink =
                TensorAccessor(sink_args, get_common_arg_val<uint32_t>(ds::writer_common_arg::ATTENTION_SINK_ADDRESS));
            experimental::CB sink_cb(cb_attention_sink), scratch(cb_sink_scratch);
            // The DRAM page is padded to the read alignment, so reading the rounded-up size stays in the buffer.
            constexpr uint32_t sink_bytes = (H * sizeof(uint16_t) + NOC_DRAM_READ_ALIGNMENT_BYTES - 1) /
                                            NOC_DRAM_READ_ALIGNMENT_BYTES * NOC_DRAM_READ_ALIGNMENT_BYTES;
            constexpr uint32_t tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);
            constexpr uint32_t faces_per_row = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
            constexpr uint32_t sink_tiles = Sqt;
            sink_cb.reserve_back(sink_tiles);
            scratch.reserve_back(1);
            noc.async_write_zeros(sink_cb, sink_tiles * tile_bytes);
            // [1, 1, 1, H] holds all heads in one row: one scalar per head into column 0 of the compute tiles.
            noc.async_read(sink, scratch, sink_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            noc.write_zeros_l1_barrier();
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_cb.get_write_ptr());
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t row = h % tt::constants::TILE_HEIGHT;
                const uint32_t offset = (h / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HW +
                                        (row / tt::constants::FACE_HEIGHT) * faces_per_row * tt::constants::FACE_HW +
                                        (row % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
                dst[offset] = src[h];
            }
            sink_cb.push_back(sink_tiles);  // reused by every token on this core
        }
    }

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        bool last = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t base, split, dst_l1;
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = request[sparse_sdpa::gather_request::BASE];
                split = request[sparse_sdpa::gather_request::SPLIT];
                last = request[sparse_sdpa::gather_request::IS_LAST] != 0;
                dst_l1 = request[sparse_sdpa::gather_request::DST_L1];
            }
            kreq_cb.pop_front(1);
            ds::gather_rows<ds::K_TRID_RING>(noc, window, cache, dst_l1, list, base, 0, split, k_row_bytes);
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }
        out_cb.wait_front(block_tiles);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(out_cb, out, out_row_bytes, {.offset_bytes = h * out_row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}

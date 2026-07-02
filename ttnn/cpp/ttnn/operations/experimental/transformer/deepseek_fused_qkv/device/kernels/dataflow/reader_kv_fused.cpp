// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Reader for the KV compute path (correctness-first v1, one KV core).
//   - hidden [1, D]  -> in0_cb (Kt tiles, resident for the whole matmul)
//   - Wkv    [K, N]  -> in1_cb, streamed in K-blocks per N-subblock in the exact order the
//                       compute kernel consumes: for ns, for kb, for kk, for w.
//   - kv_norm_w [N]  -> gain_cb (Nt tiles)
//   - cos / sin [Rd] -> cos_cb / sin_cb (rope_Wt tiles, single tile-row)
//   - trans_mat      -> trans_mat_cb (1 tile)
//   - reduce scaler (1/Dh) generated in-place into scaler_cb.
//
// hidden / gains / cos / sin / trans_mat are DRAM-interleaved; Wkv is DRAM WIDTH_SHARDED. Both
// are read uniformly by logical tile (page) index via TensorAccessor.

namespace {
// Fill a single tile with the reduce scaler (bf16 hi-half) in the layout reduce_tile expects:
// the first column of each of the 4 faces (16 entries per face).
FORCE_INLINE void generate_reduce_scaler(uint32_t cb_id, uint16_t scaler) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    if (scaler != 0) {
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t idx = k << 8;
            for (uint32_t j = 0; j < 16; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }
    cb_push_back(cb_id, 1);
}
}  // namespace

void kernel_main() {
    uint32_t argrt = 0;
    const uint32_t hidden_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t wkv_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t gain_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t gain_cb = get_compile_time_arg_val(2);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(3);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(4);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(5);
    constexpr uint32_t scaler_cb = get_compile_time_arg_val(6);
    constexpr uint32_t Kt = get_compile_time_arg_val(7);
    constexpr uint32_t Nt = get_compile_time_arg_val(8);
    constexpr uint32_t Nt_full = get_compile_time_arg_val(9);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t num_kb = get_compile_time_arg_val(12);
    constexpr uint32_t num_nsub = get_compile_time_arg_val(13);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(14);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(15);

    constexpr auto hidden_args = TensorAccessorArgs<16>();
    constexpr auto wkv_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
    constexpr auto gain_args = TensorAccessorArgs<wkv_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<gain_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;

    Noc noc;
    CircularBuffer in0_cb_obj(in0_cb);
    CircularBuffer in1_cb_obj(in1_cb);
    CircularBuffer gain_cb_obj(gain_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer trans_mat_cb_obj(trans_mat_cb);

    const uint32_t in0_tile_bytes = get_tile_size(in0_cb);
    const uint32_t in1_tile_bytes = get_tile_size(in1_cb);
    const uint32_t gain_tile_bytes = get_tile_size(gain_cb);
    const uint32_t cos_tile_bytes = get_tile_size(cos_cb);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb);
    const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb);

    const auto s_hidden = TensorAccessor(hidden_args, hidden_addr);
    const auto s_wkv = TensorAccessor(wkv_args, wkv_addr);
    const auto s_gain = TensorAccessor(gain_args, gain_addr);
    const auto s_cos = TensorAccessor(cos_args, cos_addr);
    const auto s_sin = TensorAccessor(sin_args, sin_addr);
    const auto s_trans_mat = TensorAccessor(trans_mat_args, trans_mat_addr);

    // reduce scaler (1/Dh); only the bf16 hi-half is used.
    generate_reduce_scaler(scaler_cb, static_cast<uint16_t>(scaler_bits >> 16));

    // hidden: Kt tiles, resident for the whole matmul.
    in0_cb_obj.reserve_back(Kt);
    uint32_t in0_l1 = in0_cb_obj.get_write_ptr();
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        noc.async_read(s_hidden, CoreLocalMem<uint32_t>(in0_l1), in0_tile_bytes, {.page_id = kt}, {});
        in0_l1 += in0_tile_bytes;
    }

    // gain: Nt tiles.
    gain_cb_obj.reserve_back(Nt);
    uint32_t gain_l1 = gain_cb_obj.get_write_ptr();
    for (uint32_t nt = 0; nt < Nt; ++nt) {
        noc.async_read(s_gain, CoreLocalMem<uint32_t>(gain_l1), gain_tile_bytes, {.page_id = nt}, {});
        gain_l1 += gain_tile_bytes;
    }

    // cos / sin: the single rope tile-row.
    cos_cb_obj.reserve_back(rope_Wt);
    sin_cb_obj.reserve_back(rope_Wt);
    uint32_t cos_l1 = cos_cb_obj.get_write_ptr();
    uint32_t sin_l1 = sin_cb_obj.get_write_ptr();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        noc.async_read(s_cos, CoreLocalMem<uint32_t>(cos_l1), cos_tile_bytes, {.page_id = j}, {});
        noc.async_read(s_sin, CoreLocalMem<uint32_t>(sin_l1), sin_tile_bytes, {.page_id = j}, {});
        cos_l1 += cos_tile_bytes;
        sin_l1 += sin_tile_bytes;
    }

    // trans_mat: single replicated tile.
    trans_mat_cb_obj.reserve_back(onetile);
    noc.async_read(
        s_trans_mat,
        CoreLocalMem<uint32_t>(trans_mat_cb_obj.get_write_ptr()),
        trans_mat_tile_bytes,
        {.page_id = 0},
        {});

    noc.async_read_barrier();
    in0_cb_obj.push_back(Kt);
    gain_cb_obj.push_back(Nt);
    cos_cb_obj.push_back(rope_Wt);
    sin_cb_obj.push_back(rope_Wt);
    trans_mat_cb_obj.push_back(onetile);

    // Wkv: stream K-blocks per N-subblock, matching the compute consume order.
    const uint32_t blk = in0_block_w * subblock_w;
    for (uint32_t ns = 0; ns < num_nsub; ++ns) {
        for (uint32_t kb = 0; kb < num_kb; ++kb) {
            in1_cb_obj.reserve_back(blk);
            uint32_t w_l1 = in1_cb_obj.get_write_ptr();
            for (uint32_t kk = 0; kk < in0_block_w; ++kk) {
                const uint32_t k = kb * in0_block_w + kk;
                for (uint32_t w = 0; w < subblock_w; ++w) {
                    const uint32_t n = ns * subblock_w + w;
                    const uint32_t page = k * Nt_full + n;
                    noc.async_read(s_wkv, CoreLocalMem<uint32_t>(w_l1), in1_tile_bytes, {.page_id = page}, {});
                    w_l1 += in1_tile_bytes;
                }
            }
            noc.async_read_barrier();
            in1_cb_obj.push_back(blk);
        }
    }
}

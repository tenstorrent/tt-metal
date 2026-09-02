// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

namespace {

FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

// The same reader is compiled three times with a role:
//   0: collapse cores [0,7]
//   1: post core 8
//   2: comb core 9
//
// Core 0 owns the one-core width-sharded fused_w. It reads the tile once and
// multicasts it into the identically-addressed CB on the other nine cores.
void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fused_w = get_compile_time_arg_val(1);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pre_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pre_bias = get_compile_time_arg_val(4);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(5);
    constexpr uint32_t cb_post_w = get_compile_time_arg_val(6);
    constexpr uint32_t cb_post_bias = get_compile_time_arg_val(7);
    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(8);
    constexpr uint32_t cb_comb_bias = get_compile_time_arg_val(9);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(10);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(11);
    constexpr uint32_t cb_eps_mask = get_compile_time_arg_val(12);
    constexpr uint32_t hidden_tiles_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t num_streams = get_compile_time_arg_val(14);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(15);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(16);
    constexpr uint32_t mcast_start_x = get_compile_time_arg_val(17);
    constexpr uint32_t mcast_start_y = get_compile_time_arg_val(18);
    constexpr uint32_t mcast_end_x = get_compile_time_arg_val(19);
    constexpr uint32_t mcast_end_y = get_compile_time_arg_val(20);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(21);
    constexpr uint32_t receiver_ready_sem_id = get_compile_time_arg_val(22);
    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(23);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(24);

    constexpr auto fused_w_args = TensorAccessorArgs<25>();
    constexpr auto pre_bias_args = TensorAccessorArgs<fused_w_args.next_compile_time_args_offset()>();
    constexpr auto post_bias_args = TensorAccessorArgs<pre_bias_args.next_compile_time_args_offset()>();
    constexpr auto hidden_args = TensorAccessorArgs<post_bias_args.next_compile_time_args_offset()>();
    constexpr auto comb_bias_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();

    const uint32_t fused_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t pre_bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t hidden_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_bias_addr = get_arg_val<uint32_t>(4);
    const bool is_source = get_arg_val<uint32_t>(5) != 0;

    const auto fused_w = TensorAccessor(fused_w_args, fused_w_addr);
    const auto pre_bias = TensorAccessor(pre_bias_args, pre_bias_addr);
    const auto post_bias = TensorAccessor(post_bias_args, post_bias_addr);
    const auto comb_bias = TensorAccessor(comb_bias_args, comb_bias_addr);

    CircularBuffer cb_fw(cb_fused_w);
    Semaphore<> data_ready(ready_sem_id);
    Semaphore<> receiver_ready(receiver_ready_sem_id);
    Noc noc;

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_size_bytes = cb_fw.get_tile_size();
    const uint32_t tile_elems = tile_size_bytes / 2u;

    // Collapse cores can publish the local hidden shard before fused_w arrives.
    if constexpr (role == 0) {
        CircularBuffer cb_h(cb_hidden);
        cb_h.reserve_back(hidden_tiles_per_core);
        cb_h.push_back(hidden_tiles_per_core);
    }

    cb_fw.reserve_back(one_tile);
    const uint32_t fw_l1_addr = cb_fw.get_write_ptr();
    if (is_source) {
        noc.async_read(fused_w, cb_fw, tile_size_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();

        if constexpr (num_receivers > 0) {
            receiver_ready.down(num_receivers);
            uint32_t dst_start_x = mcast_start_x;
            uint32_t dst_end_x = mcast_end_x;
            if (noc_index == 1) {
                std::swap(dst_start_x, dst_end_x);
            }
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(fw_l1_addr),
                MulticastEndpoint{},
                tile_size_bytes,
                num_receivers,
                {},
                {.noc_x_start = dst_start_x,
                 .noc_y_start = mcast_start_y,
                 .noc_x_end = dst_end_x,
                 .noc_y_end = mcast_end_y,
                 .addr = fw_l1_addr},
                /*linked=*/false);
            noc.async_writes_flushed();
            data_ready.set(1);
            data_ready.set_multicast(
                noc, dst_start_x, mcast_start_y, dst_end_x, mcast_end_y, num_receivers, /*linked=*/false);
        }
        cb_fw.push_back(one_tile);
    } else {
        receiver_ready.up(noc, sender_noc_x, sender_noc_y, 1);
        data_ready.wait(1);
        cb_fw.push_back(one_tile);
    }

    const volatile tt_l1_ptr uint16_t* fw = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_fw.get_read_ptr());
    auto fused_w_at = [&](uint32_t k) { return fw[tile_face_index(0, k & 31u)]; };

    if constexpr (role == 0) {
        CircularBuffer cb_pw(cb_pre_w);
        CircularBuffer cb_pb(cb_pre_bias);

        cb_pb.reserve_back(one_tile);
        noc.async_read(pre_bias, cb_pb, cb_pb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_pb.push_back(one_tile);

        cb_pw.reserve_back(one_tile);
        noc.async_write_zeros(cb_pw, cb_pw.get_tile_size(), {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        auto* pw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_pw.get_write_ptr());
        for (uint32_t k = 0; k < num_streams; ++k) {
            pw[tile_face_index(0, k)] = fused_w_at(k);
        }
        cb_pw.push_back(one_tile);
    } else if constexpr (role == 1) {
        CircularBuffer cb_pw(cb_post_w);
        CircularBuffer cb_pb(cb_post_bias);

        cb_pb.reserve_back(one_tile);
        noc.async_read(post_bias, cb_pb, cb_pb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_pb.push_back(one_tile);

        cb_pw.reserve_back(one_tile);
        noc.async_write_zeros(cb_pw, cb_pw.get_tile_size(), {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        auto* pw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_pw.get_write_ptr());
        for (uint32_t k = 0; k < num_streams; ++k) {
            pw[tile_face_index(0, k)] = fused_w_at(num_streams + k);
        }
        cb_pw.push_back(one_tile);
    } else {
        CircularBuffer cb_cw(cb_comb_w);
        CircularBuffer cb_cb(cb_comb_bias);
        CircularBuffer cb_scaler_obj(cb_scaler);
        CircularBuffer cb_mask_obj(cb_mask);
        CircularBuffer cb_eps_mask_obj(cb_eps_mask);

        cb_scaler_obj.reserve_back(one_tile);
        auto* scaler = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scaler_obj.get_write_ptr());
        for (uint32_t i = 0; i < tile_elems; ++i) {
            scaler[i] = 0;
        }
        const uint16_t one = static_cast<uint16_t>(scaler_bits >> 16);
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t row = 0; row < 16; ++row) {
                scaler[face * 256u + row] = one;
            }
        }
        cb_scaler_obj.push_back(one_tile);

        auto make_mask = [&](CircularBuffer& cb, uint16_t value) {
            cb.reserve_back(one_tile);
            auto* mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
            for (uint32_t i = 0; i < tile_elems; ++i) {
                mask[i] = 0;
            }
            for (uint32_t r = 0; r < num_streams; ++r) {
                for (uint32_t c = 0; c < num_streams; ++c) {
                    mask[tile_face_index(r, c)] = value;
                }
            }
            cb.push_back(one_tile);
        };
        make_mask(cb_mask_obj, one);
        make_mask(cb_eps_mask_obj, static_cast<uint16_t>(eps_bits >> 16));

        cb_cb.reserve_back(one_tile);
        noc.async_read(comb_bias, cb_cb, cb_cb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_cb.push_back(one_tile);

        cb_cw.reserve_back(one_tile);
        noc.async_write_zeros(cb_cw, cb_cw.get_tile_size(), {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        auto* cw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_cw.get_write_ptr());
        const uint32_t comb_base = 2u * num_streams;
        for (uint32_t r = 0; r < num_streams; ++r) {
            for (uint32_t c = 0; c < num_streams; ++c) {
                cw[tile_face_index(r, c)] = fused_w_at(comb_base + r * num_streams + c);
            }
        }
        cb_cw.push_back(one_tile);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Partial-width-sharded matmul activation (in0 / A) reader.
//
// A is gathered onto every compute core via the shared handshake in gather_common.hpp. This reader
// also publishes the single resident weight (in1 / B) and, when fused_residual is set, has each
// base core NoC-read its [M_tiles x Nc_tiles] N-slice of the interleaved residual (+ publish the
// resident per-channel gate) for the compute epilogue out = residual + gate * (A @ B).
#include "gather_common.hpp"
using experimental::CircularBuffer;

void kernel_main() {
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(14);  // resident B block
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t in0_M_tiles = get_compile_time_arg_val(17);  // A height in tiles (residual M)
    // fused_residual epilogue config (residual read on base cores).
    constexpr uint32_t fused_residual = get_compile_time_arg_val(20);
    constexpr uint32_t residual_cb_index = get_compile_time_arg_val(21);
    constexpr uint32_t residual_Nc_tiles = get_compile_time_arg_val(22);
    constexpr uint32_t residual_N_tiles = get_compile_time_arg_val(23);  // residual width in tiles (page stride)
    constexpr uint32_t residual_tile_size_bytes = get_compile_time_arg_val(24);  // residual dtype tile size
    constexpr uint32_t gate_cb_index = get_compile_time_arg_val(25);             // buffer-backed gate (publish it)
    constexpr uint32_t gate_num_tiles = get_compile_time_arg_val(26);            // Nc_tiles gate tiles
    constexpr auto in0_args = TensorAccessorArgs<27>();                          // (gather reads this too)
    constexpr auto residual_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t is_base = get_arg_val<uint32_t>(4);               // 1 if this core owns an output N-slice
    const uint32_t res_n_idx = get_arg_val<uint32_t>(5);             // this base core's N-slice index
    const uint32_t residual_buffer_addr = get_arg_val<uint32_t>(6);  // interleaved residual base addr

    // in1 (B) is already resident in L1; just publish it to compute.
    CircularBuffer in1_cb(in1_cb_index);
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    // fused_residual: this base core NoC-reads its [in0_M_tiles x residual_Nc_tiles] N-slice of the
    // interleaved residual into residual_cb (page = mt*N_tiles + n_idx*Nc_tiles + nc -- identical to
    // the interleaved-output writer's scatter), so compute can add it after the gate multiply.
    if (fused_residual && is_base) {
        // gate is resident (buffer-backed); publish it to compute like in1.
        cb_reserve_back(gate_cb_index, gate_num_tiles);
        cb_push_back(gate_cb_index, gate_num_tiles);
        const auto res_acc = TensorAccessor(residual_args, residual_buffer_addr, residual_tile_size_bytes);
        const uint32_t res_num_tiles = in0_M_tiles * residual_Nc_tiles;
        cb_reserve_back(residual_cb_index, res_num_tiles);
        uint32_t res_l1_addr = get_write_ptr(residual_cb_index);
        for (uint32_t mt = 0; mt < in0_M_tiles; ++mt) {
            for (uint32_t nc = 0; nc < residual_Nc_tiles; ++nc) {
                const uint32_t page = mt * residual_N_tiles + res_n_idx * residual_Nc_tiles + nc;
                noc_async_read_tile(page, res_acc, res_l1_addr);
                res_l1_addr += residual_tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(residual_cb_index, res_num_tiles);
    }

    gather_full_a<27>();  // in0 accessor at slot 27 (after the residual config)
}

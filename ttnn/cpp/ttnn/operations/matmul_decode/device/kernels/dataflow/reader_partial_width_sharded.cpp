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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

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
    // Fused adaRMS-norm prologue config. The accessors and then the scalars live AFTER the residual
    // accessor because accessor arg blocks are variable-length and gather_full_a<27> pins in0 to 27,
    // so nothing may be inserted ahead of it.
    constexpr auto nw_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto nb_args = TensorAccessorArgs<nw_args.next_compile_time_args_offset()>();
    constexpr uint32_t NCTA = nb_args.next_compile_time_args_offset();
    constexpr uint32_t fused_rms_norm = get_compile_time_arg_val(NCTA + 0);
    constexpr uint32_t norm_has_bias = get_compile_time_arg_val(NCTA + 1);
    constexpr uint32_t nw_cb_index = get_compile_time_arg_val(NCTA + 2);
    constexpr uint32_t nb_cb_index = get_compile_time_arg_val(NCTA + 3);
    constexpr uint32_t nscaler_cb_index = get_compile_time_arg_val(NCTA + 4);
    constexpr uint32_t norm_Kc_tiles = get_compile_time_arg_val(NCTA + 5);
    constexpr uint32_t norm_tile_size_bytes = get_compile_time_arg_val(NCTA + 6);
    constexpr uint32_t norm_reduce_n = get_compile_time_arg_val(NCTA + 7);  // N == K_tiles*TILE_WIDTH

    const uint32_t is_base = get_arg_val<uint32_t>(4);               // 1 if this core owns an output N-slice
    const uint32_t res_n_idx = get_arg_val<uint32_t>(5);             // this base core's N-slice index
    const uint32_t residual_buffer_addr = get_arg_val<uint32_t>(6);  // interleaved residual base addr
    const uint32_t nw_buffer_addr = get_arg_val<uint32_t>(7);        // norm weight base addr
    const uint32_t nb_buffer_addr = get_arg_val<uint32_t>(8);        // norm bias base addr
    const uint32_t norm_k_offset = get_arg_val<uint32_t>(9);         // this core's first global K-tile

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

    // fused_rms_norm: fetch only THIS core's K-slice of the per-channel norm weight (and bias), and
    // synthesize the 1/N reduce scaler the row-sum needs to become a mean. Slice-only is what keeps the
    // fusion cheap -- the full-row alternative is 4x the reads for tiles the compute never touches.
    if constexpr (fused_rms_norm) {
        const auto nw_acc = TensorAccessor(nw_args, nw_buffer_addr, norm_tile_size_bytes);
        cb_reserve_back(nw_cb_index, norm_Kc_tiles);
        uint32_t nw_l1 = get_write_ptr(nw_cb_index);
        for (uint32_t kc = 0; kc < norm_Kc_tiles; ++kc) {
            noc_async_read_tile(norm_k_offset + kc, nw_acc, nw_l1);
            nw_l1 += norm_tile_size_bytes;
        }
        if constexpr (norm_has_bias) {
            const auto nb_acc = TensorAccessor(nb_args, nb_buffer_addr, norm_tile_size_bytes);
            cb_reserve_back(nb_cb_index, norm_Kc_tiles);
            uint32_t nb_l1 = get_write_ptr(nb_cb_index);
            for (uint32_t kc = 0; kc < norm_Kc_tiles; ++kc) {
                noc_async_read_tile(norm_k_offset + kc, nb_acc, nb_l1);
                nb_l1 += norm_tile_size_bytes;
            }
        }
        // Reduce scaler: SUM with a 1/N factor so the row-sum arrives as a mean. This is the
        // documented non-standard-scaler case for prepare_reduce_scaler, which also lays the value out
        // in the row-0 form the reduce LLK requires (a naively filled tile gives wrong results) and
        // deduces format/tile shape from the CB.
        dataflow_kernel_lib::
            prepare_reduce_scaler<nscaler_cb_index, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                1.0f / static_cast<float>(norm_reduce_n));
        noc_async_read_barrier();
        cb_push_back(nw_cb_index, norm_Kc_tiles);
        if constexpr (norm_has_bias) {
            cb_push_back(nb_cb_index, norm_Kc_tiles);
        }
    }

    gather_full_a<27>();  // in0 accessor at slot 27 (after the residual config)
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NCRISC dataflow kernel — reads gate_proj and up_proj weight tiles from DRAM
// into CB_IN1_GATE and CB_IN1_UP, then writes the computed act_out (fused
// SiLU output) tiles from L1 back to DRAM.  Single-output version.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t m_blocks_local = get_compile_time_arg_val(1);
    constexpr uint32_t n_blocks_local = get_compile_time_arg_val(2);
    constexpr uint32_t Mt_block_size = get_compile_time_arg_val(3);
    constexpr uint32_t Kt_block_size = get_compile_time_arg_val(4);
    constexpr uint32_t Nt_block_size = get_compile_time_arg_val(5);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t Nt = get_compile_time_arg_val(8);

    constexpr uint32_t gate_ta_offset = 9;
    constexpr auto gate_ta_args = TensorAccessorArgs<gate_ta_offset>();
    constexpr uint32_t up_ta_offset = gate_ta_args.next_compile_time_args_offset();
    constexpr auto up_ta_args = TensorAccessorArgs<up_ta_offset>();
    constexpr uint32_t act_out_ta_off = up_ta_args.next_compile_time_args_offset();
    constexpr auto act_out_ta_args = TensorAccessorArgs<act_out_ta_off>();

    uint32_t argidx = 0;
    const uint32_t gate_proj_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t up_proj_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t act_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(argidx++);

    const auto gate_acc = TensorAccessor(gate_ta_args, gate_proj_addr, in1_tile_size);
    const auto up_acc = TensorAccessor(up_ta_args, up_proj_addr, in1_tile_size);
    const auto act_out_acc = TensorAccessor(act_out_ta_args, act_out_addr, out_tile_size);

    constexpr uint32_t cb_in1_gate = tt::CBIndex::c_1;
    constexpr uint32_t cb_in1_up = tt::CBIndex::c_2;
    constexpr uint32_t cb_act_out = tt::CBIndex::c_6;

    constexpr uint32_t in1_block_size = Kt_block_size * Nt_block_size;
    constexpr uint32_t full_N_local = n_blocks_local * Nt_block_size;
    constexpr uint32_t full_out_tiles = Mt_block_size * full_N_local;

    for (uint32_t m = 0; m < m_blocks_local; m++) {
        uint32_t m_tile_base = m_tile_start + m * Mt_block_size;

        for (uint32_t k = 0; k < K_num_blocks; k++) {
            uint32_t k_tile_base = k * Kt_block_size;
            for (uint32_t n = 0; n < n_blocks_local; n++) {
                uint32_t n_tile_base = n_tile_start + n * Nt_block_size;

                cb_reserve_back(cb_in1_gate, in1_block_size);
                uint32_t gate_write_ptr = get_write_ptr(cb_in1_gate);
                for (uint32_t kt = 0; kt < Kt_block_size; kt++) {
                    for (uint32_t nt = 0; nt < Nt_block_size; nt++) {
                        uint32_t tile_id = (k_tile_base + kt) * Nt + (n_tile_base + nt);
                        noc_async_read_page(tile_id, gate_acc, gate_write_ptr);
                        gate_write_ptr += in1_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_in1_gate, in1_block_size);

                cb_reserve_back(cb_in1_up, in1_block_size);
                uint32_t up_write_ptr = get_write_ptr(cb_in1_up);
                for (uint32_t kt = 0; kt < Kt_block_size; kt++) {
                    for (uint32_t nt = 0; nt < Nt_block_size; nt++) {
                        uint32_t tile_id = (k_tile_base + kt) * Nt + (n_tile_base + nt);
                        noc_async_read_page(tile_id, up_acc, up_write_ptr);
                        up_write_ptr += in1_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_in1_up, in1_block_size);
            }
        }

        cb_wait_front(cb_act_out, full_out_tiles);
        uint32_t act_read_ptr = get_read_ptr(cb_act_out);
        for (uint32_t mt = 0; mt < Mt_block_size; mt++) {
            for (uint32_t nt = 0; nt < full_N_local; nt++) {
                uint32_t tile_id = (m_tile_base + mt) * Nt + (n_tile_start + nt);
                noc_async_write_page(tile_id, act_out_acc, act_read_ptr);
                act_read_ptr += out_tile_size;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_act_out, full_out_tiles);
    }
}

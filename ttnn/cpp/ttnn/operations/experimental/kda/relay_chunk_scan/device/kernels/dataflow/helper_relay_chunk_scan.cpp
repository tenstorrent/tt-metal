// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/gate_matmul_dataflow.hpp"

constexpr uint32_t cb_gin0 = get_compile_time_arg_val(0);
constexpr uint32_t cb_gin1 = get_compile_time_arg_val(1);
constexpr uint32_t cb_gate = get_compile_time_arg_val(2);
constexpr uint32_t kt = get_compile_time_arg_val(3);
constexpr uint32_t core_chunks = get_compile_time_arg_val(4);
constexpr uint32_t gate_kb = get_compile_time_arg_val(5);
constexpr uint32_t gate_k = get_compile_time_arg_val(6);
constexpr uint32_t local_k = get_compile_time_arg_val(7);
constexpr uint32_t gate_columns = get_compile_time_arg_val(8);
constexpr uint32_t heads = get_compile_time_arg_val(9);
constexpr uint32_t num_chunks = get_compile_time_arg_val(10);
constexpr uint32_t cores_per_head = get_compile_time_arg_val(11);
constexpr uint32_t gate_groups = get_compile_time_arg_val(12);
constexpr uint32_t gate_table = get_compile_time_arg_val(13);
constexpr auto gate_input_args = TensorAccessorArgs<14>();
constexpr auto gate_local_args = TensorAccessorArgs<gate_input_args.next_compile_time_args_offset()>();
constexpr auto gate_weight_args = TensorAccessorArgs<gate_local_args.next_compile_time_args_offset()>();

// A head's helper forms g for the chunks of the head's first core, which runs the relay's hop instead: it takes the
// rows of x of those chunks from the group of the helpers of the head's batch and the head's columns of W from the
// group of the head's cores (gate_matmul_dataflow.hpp), then writes the g compute forms into the head core's cb_gate
// (at the same address) and counts it there.
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t head_xy = get_arg_val<uint32_t>(1);
    const uint32_t tp_rank = get_common_arg_val<uint32_t>(3);
    const uint32_t blocks = gate_k / gate_kb;

    const auto input = TensorAccessor(gate_input_args, get_common_arg_val<uint32_t>(0));
    const auto local = TensorAccessor(gate_local_args, get_common_arg_val<uint32_t>(1));
    const uint32_t x_tile = get_tile_size(cb_gin0);
    const uint32_t first_row = head / heads * num_chunks;
    const uint32_t local_begin = tp_rank * local_k;
    const auto read_x = [&](uint32_t b, uint32_t slot) {
        for (uint32_t m = 0; m < core_chunks; ++m) {
            for (uint32_t i = 0; i < gate_kb; ++i) {
                const uint32_t k = b * gate_kb + i;
                const uint32_t destination = slot + (m * gate_kb + i) * x_tile;
                if (k - local_begin < local_k) {
                    noc_async_read_page((first_row + m) * local_k + k - local_begin, local, destination);
                } else {
                    noc_async_read_page((first_row + m) * gate_k + k, input, destination);
                }
            }
        }
    };
    const auto weight = TensorAccessor(gate_weight_args, get_common_arg_val<uint32_t>(2));
    const uint32_t w_tile = get_tile_size(cb_gin1);
    const uint32_t first_column = head % heads * kt;
    const auto read_w = [&](uint32_t b, uint32_t slot) {
        for (uint32_t i = 0; i < gate_kb; ++i) {
            for (uint32_t j = 0; j < kt; ++j) {
                noc_async_read_page(
                    (b * gate_kb + i) * gate_columns + first_column + j, weight, slot + (i * kt + j) * w_tile);
            }
        }
    };
    kda_gate::Share x(
        gate_table, head / heads * cores_per_head, cb_gin0, core_chunks * gate_kb, core_chunks * gate_kb * x_tile, 4);
    kda_gate::Share w(gate_table, gate_groups + head, cb_gin1, gate_kb * kt, gate_kb * kt * w_tile, 6);
    // Both operands on this one RISC, each as far as its group lets it.
    bool x_done = false;
    bool w_done = false;
    while (!(x_done && w_done)) {
        x_done = x.step(blocks, read_x);
        w_done = w.step(blocks, read_w);
    }
    x.finish(blocks);
    w.finish(blocks);

    cb_wait_front(cb_gate, core_chunks * kt);
    const uint32_t gate = get_read_ptr(cb_gate);
    noc_async_write(gate, kda_gate::noc_address(head_xy, gate), core_chunks * kt * get_tile_size(cb_gate));
    noc_async_write_barrier();
    noc_semaphore_inc(kda_gate::noc_address(head_xy, get_common_arg_val<uint32_t>(8)), 1);
    cb_pop_front(cb_gate, core_chunks * kt);
    noc_async_atomic_barrier();
}

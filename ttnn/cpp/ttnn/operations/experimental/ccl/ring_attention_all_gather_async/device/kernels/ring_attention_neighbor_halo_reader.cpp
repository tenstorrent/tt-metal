// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ring_attention_prefetch_utils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

constexpr uint32_t my_ring_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(3);
constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_inputs = get_compile_time_arg_val(5);
constexpr uint32_t page_size_base_idx = 6;
constexpr uint32_t prefetch_packets = 4;

void kernel_main() {
    constexpr auto input_accessor_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();

    uint32_t arg_idx = 0;
    const size_t incoming_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_stride_pages;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_start;
    std::array<uint32_t, num_inputs> input_tile_end;
    std::array<uint32_t, num_inputs> input_batch_base;
    for (uint32_t input = 0; input < num_inputs; ++input) {
        input_stride_pages[input] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_start[input] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_end[input] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_base[input] = get_arg_val<uint32_t>(arg_idx++);
    }

    auto input_accessors_tuple = make_tensor_accessor_tuple(input_accessor_args, arg_idx);
    arg_idx += num_inputs;
    auto input_accessors = make_abstract_tensor_accessor_wrappers(input_accessors_tuple);

    OpSignaler op_signaler(arg_idx);

    Noc noc;
    CircularBuffer cb_output(cb_output_id);
    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;
    for (uint32_t input = 0; input < num_inputs; ++input) {
        for (uint32_t bh = 0; bh < input_batch_head_count[input]; ++bh) {
            uint32_t tiles_read = input_tile_start[input];
            prefetch_batch_read_tiles<input_page_size, packet_size_in_pages, prefetch_packets, 1>(
                noc,
                cb_output,
                tiles_read,
                input_tile_end[input],
                cb_fifo_limit,
                cb_fifo_size,
                input_accessors[input],
                [&](uint32_t tile) { return input_batch_base[input] + bh * input_stride_pages[input] + tile; });
        }
    }

    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(incoming_ready_sem), 1);
    constexpr uint32_t predecessor_ring_id = my_ring_id == 0 ? ring_size - 1 : my_ring_id - 1;
    op_signaler.synchronize_workers_and_signal_op(predecessor_ring_id);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(incoming_ready_sem), 0);
}

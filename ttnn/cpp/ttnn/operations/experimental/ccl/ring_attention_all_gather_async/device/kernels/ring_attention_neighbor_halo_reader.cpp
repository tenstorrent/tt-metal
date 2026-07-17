// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

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

template <typename Accessor>
FORCE_INLINE void read_source_range(
    const Noc& noc,
    CircularBuffer& cb_output,
    const Accessor& accessor,
    uint32_t input_stride_pages,
    uint32_t batch_head_count,
    uint32_t input_tile_start,
    uint32_t input_tile_end,
    uint32_t input_batch_base) {
    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;

    for (uint32_t bh = 0; bh < batch_head_count; ++bh) {
        uint32_t tiles_read = input_tile_start;
        while (tiles_read < input_tile_end) {
            const uint32_t remaining_pages = input_tile_end - tiles_read;
            const uint32_t remaining_packets = (remaining_pages + packet_size_in_pages - 1) / packet_size_in_pages;
            const uint32_t batch_packets = std::min(remaining_packets, prefetch_packets);
            cb_output.reserve_back(batch_packets * packet_size_in_pages);
            uint32_t l1_write_addr = cb_output.get_write_ptr();

            for (uint32_t packet = 0; packet < batch_packets; ++packet) {
                const uint32_t pages = std::min(input_tile_end - tiles_read, packet_size_in_pages);
                for (uint32_t page = 0; page < pages; ++page) {
                    if (l1_write_addr >= cb_fifo_limit) {
                        l1_write_addr -= cb_fifo_size;
                    }
                    noc.async_read(
                        accessor,
                        CoreLocalMem<uint8_t>(l1_write_addr),
                        input_page_size,
                        {.page_id = input_batch_base + bh * input_stride_pages + tiles_read++},
                        {});
                    l1_write_addr += input_page_size;
                }
                l1_write_addr += (packet_size_in_pages - pages) * input_page_size;
            }

            noc.async_read_barrier();
            for (uint32_t packet = 0; packet < batch_packets; ++packet) {
                cb_output.push_back(packet_size_in_pages);
            }
        }
    }
}

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
    for (uint32_t input = 0; input < num_inputs; ++input) {
        read_source_range(
            noc,
            cb_output,
            input_accessors[input],
            input_stride_pages[input],
            input_batch_head_count[input],
            input_tile_start[input],
            input_tile_end[input],
            input_batch_base[input]);
    }

    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(incoming_ready_sem), 1);
    constexpr uint32_t predecessor_ring_id = my_ring_id == 0 ? ring_size - 1 : my_ring_id - 1;
    op_signaler.synchronize_workers_and_signal_op(predecessor_ring_id);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(incoming_ready_sem), 0);
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);  // 2
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(6));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(7);  // 2
constexpr uint32_t num_inputs = get_compile_time_arg_val(8);
constexpr bool direction = get_compile_time_arg_val(9);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(10);

// Prefetch: batch multiple packets of DRAM reads before a single barrier.
// This keeps more reads in flight across interleaved DRAM banks, hiding latency.
// CB depth must be >= 2 * PREFETCH_PACKETS * packet_size_in_pages (see program_factory cb_num_pages).
constexpr uint32_t PREFETCH_PACKETS = 4;

// Batch-read tiles into the output CB with DRAM prefetch and FIFO wrapping.
// get_noc_addr_and_advance is called once per tile-group read; it returns the source
// NoC address and may perform per-tile side effects (e.g. row-stride tracking).
//
// Manual ring-buffer wrapping: batched reads may write across the FIFO boundary, which
// bypasses the CB's normal contiguous-write assumption (see cb_push_back). This is safe because
// noc_async_read targets raw L1 addresses and cb_push_back is called per-packet
// (packet_size_in_pages always divides cb_num_pages evenly).
template <typename AddrFn>
FORCE_INLINE void prefetch_batch_read_tiles(
    uint32_t& tiles_read,
    uint32_t tiles_to_read,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    AddrFn&& get_noc_addr_and_advance) {
    constexpr uint32_t payload_size_bytes = input_tensor_page_size * contig_pages_advanced;
    while (tiles_read < tiles_to_read) {
        uint32_t remaining_tiles = tiles_to_read - tiles_read;
        uint32_t remaining_packets = (remaining_tiles + packet_size_in_pages - 1) / packet_size_in_pages;
        uint32_t batch_packets = std::min(remaining_packets, PREFETCH_PACKETS);
        uint32_t batch_pages = batch_packets * packet_size_in_pages;

        cb_reserve_back(cb_output_id, batch_pages);
        uint32_t l1_write_addr = get_write_ptr(cb_output_id);

        for (uint32_t p = 0; p < batch_packets; p++) {
            uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
            for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                if (l1_write_addr >= cb_fifo_limit) {
                    l1_write_addr -= cb_fifo_size;
                }
                noc_async_read(get_noc_addr_and_advance(tiles_read), l1_write_addr, input_tensor_page_size);
                l1_write_addr += payload_size_bytes;
                tiles_read += contig_pages_advanced;
            }
            l1_write_addr += (packet_size_in_pages - num_pages_to_read) * input_tensor_page_size;
        }
        noc_async_read_barrier();
        for (uint32_t p = 0; p < batch_packets; p++) {
            cb_push_back(cb_output_id, packet_size_in_pages);
        }
    }
}

void kernel_main() {
    constexpr uint32_t page_size_base_idx = 11;
    constexpr auto inputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<
        num_inputs,
        std::get<num_inputs - 1>(inputs_args).next_compile_time_args_offset()>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_tensor_Wt;
    std::array<uint32_t, num_inputs> input_tensor_Ht;
    std::array<uint32_t, num_inputs> output_tensor_Wt;
    std::array<uint32_t, num_inputs> output_tensor_Ht;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_id_start;
    std::array<uint32_t, num_inputs> input_tile_id_end;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_start[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_end[input_idx] = get_arg_val<uint32_t>(arg_idx++);
    }

    auto inputs_tuple = make_tensor_accessor_tuple(inputs_args, arg_idx, page_size_base_idx);
    arg_idx += num_inputs;
    auto input_tensor_addrgens = make_abstract_tensor_accessor_wrappers(inputs_tuple);
    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx, page_size_base_idx);
    arg_idx += num_inputs;
    auto output_tensor_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;

    // Push out our local slice
    uint32_t output_tile_id_start = 0;
    // Read local slice to our buffers, before sending them over
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        uint32_t tiles_read = input_tile_id_start[input_idx];
        uint32_t tiles_to_read = input_tile_id_end[input_idx];
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
            prefetch_batch_read_tiles(tiles_read, tiles_to_read, cb_fifo_limit, cb_fifo_size, [&](uint32_t tr) {
                return input_tensor_addrgens[input_idx].get_noc_addr(output_tile_id_start + tr);
            });
            tiles_read = input_tile_id_start[input_idx];
            tiles_to_read = input_tile_id_end[input_idx];
            output_tile_id_start += input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
        }
        output_tile_id_start = 0;
    }

    uint32_t slices_received = 0;
    uint32_t slices_expected = 0;
    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1) {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_backward_direction ? num_targets_forward_direction : 0;
        } else {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_forward_direction ? num_targets_backward_direction : 0;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_backward_direction - 1;
        } else {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    while (slices_received < slices_expected) {
        // Do i expect more from the backward direction?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right, (keep in mind this differs
        // for odd/even chips)
        // Do i expect more from the forward direction?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), slices_received + 1);
        // Got it
        slices_received++;

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if constexpr (direction == 1) {
            sender_chip_id = my_chip_id + slices_received;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - slices_received;
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }

        if constexpr (fuse_op) {
            // Signal matmul to go
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
        // Direction == backward: Should I forward what I got from the left to my right?
        // In the linear case, if I have any targets to my right, always forward
        // In the ring case, if I have received on the left less than my targets on the right, forward
        // Direction == forward: Should I forward what I got from the right to my left?
        // In the linear case, if I have any targets to my left, always forward
        // In the ring case, if I have received on the right less than my targets on the left, forward
        if ((topology == Topology::Linear && writes_expected > 0) ||
            (topology == Topology::Ring && (slices_received < (writes_expected + 1)))) {
            for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
                uint32_t tiles_read = input_tile_id_start[input_idx];
                uint32_t tiles_to_read = input_tile_id_end[input_idx];

                uint32_t output_tile_id_start = 0;
                uint32_t pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                uint32_t row_offset =
                    (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                uint32_t slice_Wt = input_tensor_Wt[input_idx];
                uint32_t stride_Wt = output_tensor_Wt[input_idx];
                if (gather_dim == 3) {
                    output_tile_id_start = actual_sender_chip_id * input_tensor_Wt[input_idx];
                } else {
                    output_tile_id_start =
                        actual_sender_chip_id * input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                    prefetch_batch_read_tiles(
                        tiles_read, tiles_to_read, cb_fifo_limit, cb_fifo_size, [&](uint32_t /* tiles_read */) {
                            auto addr = output_tensor_addrgens[input_idx].get_noc_addr(
                                output_tile_id_start + row_offset + pages_read_in_row);
                            pages_read_in_row++;
                            if (pages_read_in_row >= slice_Wt) {
                                row_offset += stride_Wt;
                                pages_read_in_row = 0;
                            }
                            return addr;
                        });
                    pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                    row_offset =
                        (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                    tiles_read = input_tile_id_start[input_idx];
                    tiles_to_read = input_tile_id_end[input_idx];
                    output_tile_id_start += output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
                }
            }
        }
    }
    const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem);
    noc_inline_dw_write(dest_noc_addr, 0);
}

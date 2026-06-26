// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_named_compile_time_arg_val("my_chip_id");
constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
constexpr uint32_t cb_input_id = get_named_compile_time_arg_val("cb_input_id");  // input_tensor from reader -> compute
constexpr uint32_t cb_interm_id =
    get_named_compile_time_arg_val("cb_interm_id");  // intermediate_tensor from reader -> compute
constexpr uint32_t cb_interm2_id =
    get_named_compile_time_arg_val("cb_interm2_id");  // output_tensor from reader -> compute
constexpr uint32_t cb_reader_output_id =
    get_named_compile_time_arg_val("cb_reader_output_id");  // input_tensor from reader -> writer
constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
constexpr uint32_t input_batch_num_pages = get_named_compile_time_arg_val("input_batch_num_pages");
constexpr uint32_t output_batch_num_pages = get_named_compile_time_arg_val("output_batch_num_pages");
constexpr uint32_t input_channel_num_pages = get_named_compile_time_arg_val("input_channel_num_pages");
constexpr uint32_t output_channel_num_pages = get_named_compile_time_arg_val("output_channel_num_pages");
constexpr uint32_t input_tensor_B = get_named_compile_time_arg_val("input_tensor_B");
constexpr uint32_t input_tensor_Wt = get_named_compile_time_arg_val("input_tensor_Wt");
constexpr uint32_t slice_C = get_named_compile_time_arg_val("slice_C");
constexpr uint32_t slice_Ht = get_named_compile_time_arg_val("slice_Ht");
constexpr uint32_t slice_Wt = get_named_compile_time_arg_val("slice_Wt");
constexpr uint32_t fuse_op = get_named_compile_time_arg_val("fuse_op");
constexpr uint32_t dim = get_named_compile_time_arg_val("dim");

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t interm_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t out2_ready_sem = get_arg_val<uint32_t>(arg_idx++);  // out_ready_sem from opposite dir
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);

    constexpr auto interm_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);

    constexpr auto output_tensor_args = TensorAccessorArgs<interm_tensor_args.next_compile_time_args_offset()>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

    Noc noc_obj;
    CircularBuffer cb_input(cb_input_id);
    CircularBuffer cb_interm(cb_interm_id);
    CircularBuffer cb_interm2(cb_interm2_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);

    uint32_t sem_target = 0;
    uint32_t sem2_target = 0;

    for (uint32_t b = 0; b < input_tensor_B; ++b) {
        if constexpr (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        uint32_t batch_offset = input_batch_num_pages * b;

        // Loop over the slices, starting from the chip half-way across the ring, and working backwards
        // until we get to ourselves.
        //
        // In some iters we process a full tensor slice, and sometimes only half slice.
        // In the 1st iter we don't perform a reduction, in other iters we reduce 2 tensors.
        // In the last iter we reduce 3 tensors (local + remote from fwd device + remote from bwd device).
        // In the last iter the writer outputs to local output tensor, in other iters it sends to next chip.
        // These behaviors are controlled by a "state machine" to avoid code duplication in the loop body.
        constexpr uint32_t ring_size_by_2 = ring_size / 2;
        int slice_idx = my_chip_id + ring_size_by_2;  // start with slice belonging to device half-way across in ring
        uint32_t num_iters = ring_size_by_2 + 1;
        for (uint32_t i = 0; i < num_iters; ++i) {
            // State machine for control variables
            bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, reduce_output;
            if (i == 0) {
                even_chunks = direction;     // process the even chunks (half the tensor slice)
                odd_chunks = !direction;     // process the odd chunks (other half of tensor slice)
                reduce_even_chunks = false;  // (input_tensor + interm_tensor) or (input_tensor)
                reduce_odd_chunks = false;   // (input_tensor + interm_tensor) or (input_tensor)
                reduce_output =
                    false;  // (input_tensor + interm_tensor + output_tensor) or (input_tensor + interm_tensor)
            } else if (i == ring_size_by_2) {
                even_chunks = direction;
                odd_chunks = !direction;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                reduce_output = true;
            } else if (i == 1) {
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = direction;
                reduce_odd_chunks = !direction;
                reduce_output = false;
            } else {
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                reduce_output = false;
            }

            // below code does 'slice_idx = slice_idx % ring_size'
            if (slice_idx < 0) {
                slice_idx += ring_size;
            } else if (slice_idx >= (int)ring_size) {
                slice_idx = (uint32_t)slice_idx - ring_size;
            }

            // address incrementer for input_tensor and interm_tensor
            uint32_t input_tile_id_start, interm_tile_id_start;
            if constexpr (dim == 3) {
                input_tile_id_start = slice_idx * slice_Wt + batch_offset;
                interm_tile_id_start = slice_idx * slice_Wt;
            } else if constexpr (dim == 2) {
                input_tile_id_start = slice_idx * slice_Ht * slice_Wt + batch_offset;
                interm_tile_id_start = slice_idx * slice_Ht * slice_Wt;
            } else if constexpr (dim == 1) {
                input_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt + batch_offset;
                interm_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt;
            } else {
                ASSERT(false);
            }
            uint32_t input_pages_read_in_row = start_pages_read_in_row,
                     interm_pages_read_in_row = start_pages_read_in_row;
            uint32_t input_row_offset = start_row_offset, interm_row_offset = start_row_offset;
            auto get_next_input_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                ++input_pages_read_in_row;
                if (input_pages_read_in_row == slice_Wt) {
                    input_row_offset += input_tensor_Wt;
                    input_pages_read_in_row -= slice_Wt;
                }
                return tile_id;
            };
            auto get_next_interm_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = interm_tile_id_start + interm_row_offset + interm_pages_read_in_row;
                ++interm_pages_read_in_row;
                if (interm_pages_read_in_row == slice_Wt) {
                    interm_row_offset += input_tensor_Wt;
                    interm_pages_read_in_row -= slice_Wt;
                }
                return tile_id;
            };

            // address incrementer for output_tensor
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            uint32_t output_tiles_read = start_tiles_read;
            auto get_next_output_tile_id = [&]() -> uint32_t { return output_tile_id_start + (output_tiles_read++); };

            uint32_t chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                // reset addr counters
                input_pages_read_in_row = interm_pages_read_in_row = start_pages_read_in_row;
                input_row_offset = interm_row_offset = start_row_offset;
                output_tiles_read = start_tiles_read;
                uint32_t tiles_read = start_tiles_read;
                uint32_t total_tiles_to_read = start_tiles_to_read;

                /**
                 * Interleave forward and backward ring reads
                 * forward handles even chunks, backward handles odd chunks (1 chunk = tile_granularity tiles)
                 * after ring_size-1 steps, we've transferred all tiles.
                 *
                 * Chunk forward/backward parity is fixed, independent of chunk count or distribution to workers
                 */
                while (tiles_read < total_tiles_to_read) {
                    const auto [is_even_chunk, tiles_to_read] =
                        reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);

                    if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                        // Skip this chunk
                        tiles_read += tiles_to_read;
                        for (uint32_t k = 0; k < tiles_to_read; ++k) {
                            get_next_input_tile_id();
                            get_next_interm_tile_id();
                            get_next_output_tile_id();
                        }
                    } else {
                        const bool reduce_interm =
                            (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);
                        CircularBuffer& cb_in = reduce_interm ? cb_input : cb_reader_output;  // to compute or writer

                        // Wait for intermediate_tensor data to be available
                        if (reduce_interm) {
                            if (chunk_count == 0) {
                                noc_semaphore_wait_min(
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                                ++sem_target;
                                if (reduce_output) {
                                    noc_semaphore_wait_min(
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out2_ready_sem),
                                        sem2_target + 1);
                                    ++sem2_target;
                                }
                            }
                            chunk_count = (chunk_count == chunks_per_sync - 1) ? 0 : (chunk_count + 1);
                        }

                        cb_in.reserve_back(tile_granularity);
                        uint32_t l1_write_offset = 0;
                        uint32_t interm_l1_write_offset = 0, interm2_l1_write_offset = 0;
                        if (reduce_interm) {
                            cb_interm.reserve_back(tile_granularity);
                            if (reduce_output) {
                                cb_interm2.reserve_back(tile_granularity);
                            }
                        }
                        for (uint32_t j = 0; j < tiles_to_read; ++j) {
                            auto input_tile_id = get_next_input_tile_id();
                            auto interm_tile_id = get_next_interm_tile_id();
                            auto output_tile_id = get_next_output_tile_id();

                            // input_tensor from reader -> compute or writer
                            noc_obj.async_read(
                                input_tensor_accessor,
                                cb_in,
                                page_size,
                                {.page_id = input_tile_id},
                                {.offset_bytes = l1_write_offset});
                            l1_write_offset += page_size;

                            if (reduce_interm) {
                                // interm_tensor from reader -> compute
                                noc_obj.async_read(
                                    interm_tensor_accessor,
                                    cb_interm,
                                    page_size,
                                    {.page_id = interm_tile_id},
                                    {.offset_bytes = interm_l1_write_offset});
                                interm_l1_write_offset += page_size;

                                if (reduce_output) {
                                    // output_tensor from reader -> compute
                                    noc_obj.async_read(
                                        output_tensor_accessor,
                                        cb_interm2,
                                        page_size,
                                        {.page_id = output_tile_id},
                                        {.offset_bytes = interm2_l1_write_offset});
                                    interm2_l1_write_offset += page_size;
                                }
                            }
                        }
                        tiles_read += tiles_to_read;
                        noc_obj.async_read_barrier();
                        cb_in.push_back(tile_granularity);
                        if (reduce_interm) {
                            cb_interm.push_back(tile_granularity);

                            if (reduce_output) {
                                cb_interm2.push_back(tile_granularity);
                            }
                        }
                    }  // if skip or process
                }  // while total_tiles_to_read

                input_tile_id_start += input_channel_num_pages;
                interm_tile_id_start += input_channel_num_pages;
                output_tile_id_start += output_channel_num_pages;
            }

            // Next slice idx
            slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
        }

        // Reset the semaphore before the next batch
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out2_ready_sem), 0);
        sem_target = 0;
        sem2_target = 0;
    }
}

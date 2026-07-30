// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
namespace sched = ttnn::ccl::schedule;

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

    uint32_t sem_target = 0;
    uint32_t sem2_target = 0;

    // The ring schedule — slice walk, per-step flags, even/odd chunk split and the tile-id walkers —
    // comes from the shared header, so this reader, the compute kernel and the writer are driven by
    // ONE definition. Previously each carried its own copy; the reader's and the compute kernel's
    // step state machines were byte-identical, and a divergence between any two of them shows up as
    // a CB-wait deadlock or silently mis-reduced output.
    static_assert(
        sched::is_supported_scatter_dim(dim), "ring reduce-scatter supports dim 1, 2 or 3 (dim 0 is dim_zero_*)");
    sched::RingRsSchedule schedule(
        ring_size, input_tensor_B, slice_C, tile_granularity, start_tiles_read, start_tiles_to_read, direction);
    sched::SliceRowWalker input_walker(slice_Wt, input_tensor_Wt);
    sched::SliceRowWalker interm_walker(slice_Wt, input_tensor_Wt);
    sched::SequentialTileWalker output_walker;

    while (schedule.next_batch()) {
        const uint32_t b = schedule.batch_idx();
        if constexpr (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        uint32_t batch_offset = input_batch_num_pages * b;
        // Per-batch: every batch restarts the ring walk at the same first slice.
        sched::RingSliceCursor slice_cursor(my_chip_id, ring_size, direction);

        // Walk the slices, starting from the chip half-way across the ring and working back to
        // ourselves. Some steps process a full tensor slice, some only half. Step 0 performs no
        // reduction; middle steps reduce 2 tensors; the last reduces 3 (local + remote from the fwd
        // device + remote from the bwd device) and the writer lands it in the local output tensor
        // rather than sending it on. All of that now comes from schedule.flags().
        while (schedule.next_step()) {
            const bool reduce_output = schedule.flags().reduce_output;
            const uint32_t slice_idx = slice_cursor.wrap();

            const uint32_t slice_offset = sched::slice_tile_offset(dim, slice_idx, slice_C, slice_Ht, slice_Wt);
            input_walker.set_base(slice_offset + batch_offset);
            interm_walker.set_base(slice_offset);
            output_walker.set_base(b * output_batch_num_pages);

            uint32_t chunk_count = 0;
            while (schedule.next_channel()) {
                // reset addr counters
                input_walker.reset_offsets(start_pages_read_in_row, start_row_offset);
                interm_walker.reset_offsets(start_pages_read_in_row, start_row_offset);
                output_walker.reset_offsets(start_tiles_read);

                /**
                 * Interleave forward and backward ring reads
                 * forward handles even chunks, backward handles odd chunks (1 chunk = tile_granularity tiles)
                 * after ring_size-1 steps, we've transferred all tiles
                 */
                while (schedule.next_chunk()) {
                    const uint32_t tiles_to_read = schedule.tiles_this_chunk();

                    if (schedule.skip()) {
                        // Not this worker's parity this step: keep the walkers in step with the
                        // schedule and move on.
                        input_walker.advance(tiles_to_read);
                        interm_walker.advance(tiles_to_read);
                        output_walker.advance(tiles_to_read);
                        continue;
                    }

                    const bool reduce_interm = schedule.reduce_interm();
                    const uint32_t cb_in = reduce_interm ? cb_input_id : cb_reader_output_id;  // to compute or writer

                    // Wait for intermediate_tensor data to be available
                    if (reduce_interm) {
                        if (chunk_count == 0) {
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                            ++sem_target;
                            if (reduce_output) {
                                noc_semaphore_wait_min(
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out2_ready_sem), sem2_target + 1);
                                ++sem2_target;
                            }
                        }
                        chunk_count = (chunk_count == chunks_per_sync - 1) ? 0 : (chunk_count + 1);
                    }

                    cb_reserve_back(cb_in, tile_granularity);
                    uint32_t l1_write_addr = get_write_ptr(cb_in);
                    uint32_t interm_l1_write_addr, interm2_l1_write_addr;
                    if (reduce_interm) {
                        cb_reserve_back(cb_interm_id, tile_granularity);
                        interm_l1_write_addr = get_write_ptr(cb_interm_id);
                        if (reduce_output) {
                            cb_reserve_back(cb_interm2_id, tile_granularity);
                            interm2_l1_write_addr = get_write_ptr(cb_interm2_id);
                        }
                    }
                    for (uint32_t j = 0; j < tiles_to_read; ++j) {
                        auto input_tile_id = input_walker.next();
                        auto interm_tile_id = interm_walker.next();
                        auto output_tile_id = output_walker.next();

                        // input_tensor from reader -> compute or writer
                        uint64_t noc_read_addr = input_tensor_accessor.get_noc_addr(input_tile_id);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;

                        if (reduce_interm) {
                            // interm_tensor from reader -> compute
                            uint64_t interm_noc_read_addr = interm_tensor_accessor.get_noc_addr(interm_tile_id);
                            noc_async_read(interm_noc_read_addr, interm_l1_write_addr, page_size);
                            interm_l1_write_addr += page_size;

                            if (reduce_output) {
                                // output_tensor from reader -> compute
                                uint64_t output_noc_read_addr = output_tensor_accessor.get_noc_addr(output_tile_id);
                                noc_async_read(output_noc_read_addr, interm2_l1_write_addr, page_size);
                                interm2_l1_write_addr += page_size;
                            }
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_in, tile_granularity);
                    if (reduce_interm) {
                        cb_push_back(cb_interm_id, tile_granularity);

                        if (reduce_output) {
                            cb_push_back(cb_interm2_id, tile_granularity);
                        }
                    }
                }  // while chunks

                input_walker.bump_base(input_channel_num_pages);
                interm_walker.bump_base(input_channel_num_pages);
                output_walker.bump_base(output_channel_num_pages);
            }

            // Next slice idx
            slice_cursor.advance();
        }

        // Reset the semaphore before the next batch
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out2_ready_sem), 0);
        sem_target = 0;
        sem2_target = 0;
    }
}

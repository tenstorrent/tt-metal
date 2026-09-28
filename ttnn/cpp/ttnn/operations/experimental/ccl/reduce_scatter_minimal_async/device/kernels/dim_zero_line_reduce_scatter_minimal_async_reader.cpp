// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;
namespace sched = ttnn::ccl::schedule;  // the line schedule shared with the writer + compute kernel

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t output_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t batch_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t slice_B = get_compile_time_arg_val(10);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(11);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    Semaphore<> fwd_bwd_sem(get_arg_val<uint32_t>(arg_idx++));
    const bool is_forward = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_device_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_targets_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const bool do_final_reduction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 12;

#ifdef INPUT_IS_SHARDED
    constexpr uint32_t ct_offset_one = 7;

    using input_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx),       // Memory layout
        get_compile_time_arg_val(ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(ct_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + 6)>;  // pages_per_shard_y

    const auto [input_mapping_table, input_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<input_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<input_tensor_shard_info> input_tensor_addrgen = {
        .bank_base_address = input_tensor_address, .shard_array = input_mapping_table};

    arg_idx += input_rt_increment;
#else
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset_one = input_tensor_args.num_compile_time_args();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address);
#endif

#ifdef INTERMEDIATE_IS_SHARDED
    constexpr uint32_t ct_offset_two = 7;

    constexpr uint32_t inter_start_ct_idx = ct_idx + ct_offset_one;
    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(inter_start_ct_idx),       // Memory layout
        get_compile_time_arg_val(inter_start_ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(inter_start_ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(inter_start_ct_idx + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(inter_start_ct_idx + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(inter_start_ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(inter_start_ct_idx + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_tensor_addrgen = {
        .bank_base_address = intermediate_tensor_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;
#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx + ct_offset_one>();
    constexpr uint32_t ct_offset_two = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address);
#endif

#ifdef OUTPUT_IS_SHARDED
    constexpr uint32_t output_start_ct_idx = ct_idx + ct_offset_one + ct_offset_two;
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(output_start_ct_idx),       // Memory layout
        get_compile_time_arg_val(output_start_ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(output_start_ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(output_start_ct_idx + 3),   // The number of pages in each sharding row not including
                                                             // padding pages
        get_compile_time_arg_val(output_start_ct_idx + 4),   // This defines times when contiguous pages can't be
                                                             // calculated
        get_compile_time_arg_val(output_start_ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(output_start_ct_idx + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_tensor_addrgen = {
        .bank_base_address = output_tensor_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset_one + ct_offset_two>();
    auto output_tensor_addrgen = TensorAccessor(output_tensor_args, output_tensor_address);
#endif

    Noc noc_obj;
    CircularBuffer cb_input(cb_input_id);
    CircularBuffer cb_intermediate(cb_intermediate_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);

    /**
     * Intermediate buffer is double-sized (shape [2, *input_shape]) to accommodate forward and backward.
     * BWD indexes into second half of intermediate buffer.
     */
    const uint32_t intermediate_full_offset = is_forward ? 0 : input_num_pages;

    // The line schedule — the no-wrap slice sequence, the chunk boundaries (slice_B plays the
    // channel role in the dim-zero family), and the chunks-per-sync wait cadence — comes from the
    // shared header. Tile ids are dense here, so the walkers are SequentialTileWalkers.
    sched::LineSliceCursor slice_cursor(is_forward, ring_size);
    sched::LineChannelWalk walk(slice_B, tile_granularity, start_tiles_read, start_tiles_to_read);
    sched::SyncCadence cadence(chunks_per_sync);
    sched::SequentialTileWalker input_walker;
    sched::SequentialTileWalker interm_walker;

    uint32_t fwd_sync_cnt = 0;
    uint32_t sem_target = 0;

    // Iterate over the slices in the direction we are going.
    // In forwards direction, count down from slice (ring_size -1) down to (my_chip_id+1), inclusive
    // In backwards direction, count up from slice 0 to (my_chip_id-1), inclusive
    // After doing all partial reductions and send, there's a final reduction step.
    // If we are not the first device in the direction, do the final reduction.
    // If this device has both FWD and BWD neighbors, the FWD reader will do final reduction first
    // and then signal the BWD reader to do its final reduction.
    for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
        cadence.reset();
        input_walker.set_base(slice_cursor.slice() * output_num_pages);
        interm_walker.set_base(slice_cursor.slice() * output_num_pages + intermediate_full_offset);

        // First device in the direction has no incoming slices, so it forwards its input directly
        // to the writer; every other device feeds the compute kernel and reads the intermediate.
        CircularBuffer& cb_in0 = is_first_device_in_direction ? cb_reader_output : cb_input;

        walk.reset();
        while (walk.next_channel()) {
            input_walker.reset_offsets(start_tiles_read);
            interm_walker.reset_offsets(start_tiles_read);

            while (walk.next_chunk()) {
                const uint32_t num_pages_to_read = walk.tiles_this_chunk();

                if (!is_first_device_in_direction) {
                    // Wait for intermediate_tensor data to be available (once per chunks_per_sync
                    // chunks — pairs with the neighbouring writer's incs).
                    if (cadence.wait_due()) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                    }
                    cadence.advance();
                }

                cb_in0.reserve_back(tile_granularity);
                uint32_t l1_write_addr = cb_in0.get_write_ptr();
                for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                    uint64_t noc_read_addr = input_tensor_addrgen.get_noc_addr(input_walker.next());
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }

                if (!is_first_device_in_direction) {
                    // read the next intermediate slice out of intermediate buffer, and put it in
                    // the intermediate CB
                    cb_intermediate.reserve_back(tile_granularity);
                    l1_write_addr = cb_intermediate.get_write_ptr();
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint64_t noc_read_addr = intermediate_tensor_addrgen.get_noc_addr(interm_walker.next());
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;
                    }
                } else {
                    interm_walker.advance(num_pages_to_read);
                }

                noc_obj.async_read_barrier();
                cb_in0.push_back(tile_granularity);
                if (!is_first_device_in_direction) {
                    cb_intermediate.push_back(tile_granularity);
                }
            }
            input_walker.bump_base(batch_num_pages);
            interm_walker.bump_base(batch_num_pages);
        }

        slice_cursor.advance();
    }

    // Do the final reduction. Synchronize with other direction.
    if (do_final_reduction) {
        cadence.reset();

        /**
         * If two cores are doing final reduction, BWD core will accumulate output with
         * incoming BWD intermediate, using the output address generator:
         * output += intermediate. Otherwise, output = input + intermediate.
         * One shared definition of the mode split (the writer holds the other half).
         */
        const bool accumulate_output = sched::line_rs_accumulate_output(sync_with_other_direction, is_forward);

        sched::SequentialTileWalker main_walker;
        main_walker.set_base(accumulate_output ? 0 : my_chip_id * output_num_pages);
        interm_walker.set_base(my_chip_id * output_num_pages + intermediate_full_offset);

        walk.reset();
        while (walk.next_channel()) {
            main_walker.reset_offsets(start_tiles_read);
            interm_walker.reset_offsets(start_tiles_read);

            while (walk.next_chunk()) {
                // Wait for FWD writer to signal that it has done its final reduction
                if (accumulate_output) {
                    fwd_bwd_sem.wait_min(++fwd_sync_cnt);
                }

                const uint32_t num_pages_to_read = walk.tiles_this_chunk();

                cb_input.reserve_back(tile_granularity);
                uint32_t l1_write_addr = cb_input.get_write_ptr();
                for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                    const uint32_t tile_id = main_walker.next();
                    uint64_t noc_read_addr = accumulate_output ? output_tensor_addrgen.get_noc_addr(tile_id)
                                                               : input_tensor_addrgen.get_noc_addr(tile_id);
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }

                if (cadence.wait_due()) {
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                }
                cadence.advance();

                // read the next intermediate slice out of the intermediate buffer, and put it in
                // the intermediate CB
                cb_intermediate.reserve_back(tile_granularity);
                l1_write_addr = cb_intermediate.get_write_ptr();
                for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                    uint64_t noc_read_addr = intermediate_tensor_addrgen.get_noc_addr(interm_walker.next());
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }

                noc_obj.async_read_barrier();
                cb_input.push_back(tile_granularity);
                cb_intermediate.push_back(tile_granularity);
            }
            main_walker.bump_base(batch_num_pages);
            interm_walker.bump_base(batch_num_pages);
        }
    }

    // Reset my output ready semaphore
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}

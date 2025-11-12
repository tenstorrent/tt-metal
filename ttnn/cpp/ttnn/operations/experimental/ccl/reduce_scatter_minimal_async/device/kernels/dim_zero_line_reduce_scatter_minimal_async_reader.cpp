// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;

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

namespace detail {
inline bool do_accumulate_output(const bool is_forward) {
    if constexpr (sync_with_other_direction) {
        return !is_forward;
    }
    return false;
}
}  // namespace detail

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
    uint32_t fwd_bwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
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
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);
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
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address, page_size);
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
    auto output_tensor_addrgen = TensorAccessor(output_tensor_args, output_tensor_address, page_size);
#endif

    /**
     * Intermediate buffer is double-sized (shape [2, *input_shape]) to accommodate forward and backward.
     * BWD indexes into second half of intermediate buffer.
     */
    const uint32_t intermediate_full_offset = is_forward ? 0 : input_num_pages;

    uint32_t chunk_count = 0;
    uint32_t fwd_sync_cnt = 0;
    uint32_t sem_target = 0;

    int slice_idx = is_forward ? ring_size - 1 : 0;

    // Iterate over the slices in the direction we are going.
    // In forwards direction, count down from slice (ring_size -1) down to (my_chip_id+1), inclusive
    // In backwards direction, count up from slice 0 to (my_chip_id-1), inclusive
    // After doing all partial reductions and send, there's a final reduction step.
    // If we are not the first device in the direction, do the final reduction.
    // If this device has both FWD and BWD neighbors, the FWD reader will do final reduction first
    // and then signal the BWD reader to do its final reduction.
    for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
        chunk_count = 0;

        uint32_t input_tile_id_start = slice_idx * output_num_pages;
        uint32_t intermediate_tile_id_start = input_tile_id_start + intermediate_full_offset;

        if (is_first_device_in_direction) {
            // We have no incoming slices, so forward directly to writer
            uint32_t cb_in0 = cb_reader_output_id;

            for (uint32_t b = 0; b < slice_B; ++b) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_reserve_back(cb_in0, tile_granularity);
                    uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t tile_id = input_tile_id_start + tiles_read + j;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;
                    }
                    tiles_read += num_pages_to_read;

                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                }
                input_tile_id_start += batch_num_pages;
            }
        } else {
            // I have incoming slices, so write my output to compute kernel and read intermediate input
            uint32_t cb_in0 = cb_input_id;

            for (uint32_t b = 0; b < slice_B; ++b) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_reserve_back(cb_in0, tile_granularity);
                    uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t tile_id = input_tile_id_start + tiles_read + j;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;
                    }

                    if (chunk_count % chunks_per_sync == 0) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                    }
                    chunk_count++;

                    // read the next intermediate slice out of intermediate buffer, and put it in intermediate CB
                    cb_reserve_back(cb_intermediate_id, tile_granularity);
                    l1_write_addr = get_write_ptr(cb_intermediate_id);
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t tile_id = intermediate_tile_id_start + tiles_read + j;
                        uint64_t noc_read_addr = get_noc_addr(tile_id, intermediate_tensor_addrgen);
                        noc_async_read(noc_read_addr, l1_write_addr, page_size);
                        l1_write_addr += page_size;
                    }

                    tiles_read += num_pages_to_read;
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, tile_granularity);
                    cb_push_back(cb_intermediate_id, tile_granularity);
                }
                input_tile_id_start += batch_num_pages;
                intermediate_tile_id_start += batch_num_pages;
            }
        }

        // Next slice idx
        if (is_forward) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }

    // Do the final reduction. Synchronize with other direction.
    if (do_final_reduction) {
        chunk_count = 0;

        uint32_t input_tile_id_start = my_chip_id * output_num_pages;
        uint32_t intermediate_tile_id_start = input_tile_id_start + intermediate_full_offset;
        uint32_t output_tile_id_start = 0;

        /**
         * If two cores are doing final reduction, BWD core will accumulate output with
         * incoming BWD intermediate. Use output address generator.
         * If true, output += intermediate. Otherwise, output = input + intermediate
         */
        uint32_t tile_id_start = detail::do_accumulate_output(is_forward) ? output_tile_id_start : input_tile_id_start;

        uint32_t cb_in0 = cb_input_id;
        for (uint32_t b = 0; b < slice_B; ++b) {
            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            while (tiles_read < tiles_to_read) {
                // Wait for FWD writer to signal that it has done its final reduction
                if (detail::do_accumulate_output(is_forward)) {
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_bwd_sem_addr), ++fwd_sync_cnt);
                }

                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                cb_reserve_back(cb_in0, tile_granularity);
                uint32_t l1_write_addr = get_write_ptr(cb_in0);
                for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                    uint32_t tile_id = tile_id_start + tiles_read + j;
                    uint64_t noc_read_addr = detail::do_accumulate_output(is_forward)
                                                 ? get_noc_addr(tile_id, output_tensor_addrgen)
                                                 : get_noc_addr(tile_id, input_tensor_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }

                if (chunk_count % chunks_per_sync == 0) {
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                }
                chunk_count++;

                // read the next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                cb_reserve_back(cb_intermediate_id, tile_granularity);
                l1_write_addr = get_write_ptr(cb_intermediate_id);
                for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                    uint32_t intermediate_tile_id = intermediate_tile_id_start + tiles_read + j;
                    uint64_t noc_read_addr = get_noc_addr(intermediate_tile_id, intermediate_tensor_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }

                tiles_read += num_pages_to_read;
                noc_async_read_barrier();
                cb_push_back(cb_in0, tile_granularity);
                cb_push_back(cb_intermediate_id, tile_granularity);
            }
            tile_id_start += batch_num_pages;
            intermediate_tile_id_start += batch_num_pages;
        }
    }

    // Reset my output ready semaphore
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}

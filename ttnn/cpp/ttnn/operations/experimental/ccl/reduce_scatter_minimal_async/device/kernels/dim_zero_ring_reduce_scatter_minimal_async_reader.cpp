// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
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
constexpr uint32_t cb_input_id = get_named_compile_time_arg_val("cb_input_id");
constexpr uint32_t cb_intermediate_id = get_named_compile_time_arg_val("cb_interm_id");
constexpr uint32_t cb_reader_output_id = get_named_compile_time_arg_val("cb_reader_output_id");
constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
constexpr uint32_t output_num_pages = get_named_compile_time_arg_val("output_num_pages");
constexpr uint32_t batch_num_pages = get_named_compile_time_arg_val("batch_num_pages");
constexpr uint32_t slice_B = get_named_compile_time_arg_val("slice_B");

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address);

    Noc noc_obj;
    CircularBuffer cb_input(cb_input_id);
    CircularBuffer cb_intermediate(cb_intermediate_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);

    uint32_t sem_target = 0;

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        const bool do_reduce = i != 0;
        CircularBuffer& cb_in0 = do_reduce ? cb_input : cb_reader_output;

        uint32_t actual_slice_idx;
        if (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        uint32_t tile_id_start = actual_slice_idx * output_num_pages;

        uint32_t chunk_count = 0;
        for (uint32_t b = 0; b < slice_B; ++b) {
            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            if (!direction) {
                uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                tiles_read += backwards_offset;
            }

            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                if (do_reduce && (chunk_count % chunks_per_sync == 0)) {
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                    sem_target++;
                }
                chunk_count++;

                uint32_t tiles_to_read_in_current_direction = 0;
                if (direction) {
                    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                } else {
                    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                }

                cb_in0.reserve_back(tile_granularity);
                uint32_t l1_write_offset = 0;
                for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                    uint32_t input_tile_id = tile_id_start + tiles_read + j;
                    noc_obj.async_read(
                        input_tensor_addrgen,
                        cb_in0,
                        page_size,
                        {.page_id = input_tile_id},
                        {.offset_bytes = l1_write_offset});
                    l1_write_offset += page_size;
                }

                if (do_reduce) {
                    // read next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                    cb_intermediate.reserve_back(tile_granularity);
                    uint32_t intermediate_l1_write_offset = 0;
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        uint32_t intermediate_tile_id = tile_id_start + tiles_read + j;
                        noc_obj.async_read(
                            intermediate_tensor_addrgen,
                            cb_intermediate,
                            page_size,
                            {.page_id = intermediate_tile_id},
                            {.offset_bytes = intermediate_l1_write_offset});
                        intermediate_l1_write_offset += page_size;
                    }

                    noc_obj.async_read_barrier();
                    cb_intermediate.push_back(tile_granularity);
                }

                tiles_read += tiles_to_read_in_current_direction;
                noc_obj.async_read_barrier();
                cb_in0.push_back(tile_granularity);

                // Skip the tiles going the other direction
                tiles_remaining_to_read = tiles_to_read - tiles_read;
                if (tiles_remaining_to_read > 0) {
                    uint32_t tiles_to_read_in_other_direction = 0;
                    if (!direction) {
                        tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }
                    tiles_read += tiles_to_read_in_other_direction;
                }
            }
            tile_id_start += batch_num_pages;
        }

        // Next slice idx
        if (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }

        if (do_reduce && (i == (ring_size - 1))) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
            sem_target = 0;
        }
    }
}

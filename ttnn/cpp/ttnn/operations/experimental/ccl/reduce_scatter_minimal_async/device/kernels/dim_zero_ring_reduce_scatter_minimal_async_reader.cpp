// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
namespace sched = ttnn::ccl::schedule;  // the dim-zero ring schedule shared with the writer + compute kernel

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
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address);

    uint32_t sem_target = 0;

    // The dim-zero ring schedule — the neighbour-first slice walk, the interleaved own/other chunk
    // pairing (a zero-tile own chunk still runs the full CB protocol), and the chunks-per-sync wait
    // cadence — comes from the shared header, so this reader, the compute kernel and the writer
    // walk ONE definition instead of three hand-maintained copies of the interleave.
    auto slice_cursor = sched::RingSliceCursor::starting_at(
        sched::ring_neighbour_first_slice(my_chip_id, direction), ring_size, direction);
    sched::DimZeroChunkWalk walk(slice_B, tile_granularity, start_tiles_read, start_tiles_to_read, direction);
    sched::SyncCadence cadence(chunks_per_sync);

    for (uint32_t i = 0; i < ring_size; ++i) {
        const bool do_reduce = i != 0;
        const uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;
        const uint32_t actual_slice_idx = slice_cursor.wrap();

        uint32_t tile_id_start = actual_slice_idx * output_num_pages;

        cadence.reset();
        walk.reset();
        while (walk.next_batch()) {
            while (walk.next_chunk()) {
                if (do_reduce && cadence.wait_due()) {
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), ++sem_target);
                }
                cadence.advance();

                const uint32_t tiles_this_chunk = walk.tiles_this_chunk();

                cb_reserve_back(cb_in0, tile_granularity);
                uint32_t l1_write_addr = get_write_ptr(cb_in0);
                for (uint32_t j = 0; j < tiles_this_chunk; ++j) {
                    uint32_t input_tile_id = tile_id_start + walk.position() + j;
                    uint64_t noc_read_addr = input_tensor_addrgen.get_noc_addr(input_tile_id);
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                }

                if (do_reduce) {
                    // read next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                    cb_reserve_back(cb_intermediate_id, tile_granularity);
                    uint32_t intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                    for (uint32_t j = 0; j < tiles_this_chunk; ++j) {
                        uint32_t intermediate_tile_id = tile_id_start + walk.position() + j;
                        uint64_t intermediate_noc_read_addr =
                            intermediate_tensor_addrgen.get_noc_addr(intermediate_tile_id);
                        noc_async_read(intermediate_noc_read_addr, intermediate_l1_write_addr, page_size);
                        intermediate_l1_write_addr += page_size;
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_intermediate_id, tile_granularity);
                }

                noc_async_read_barrier();
                cb_push_back(cb_in0, tile_granularity);
            }
            tile_id_start += batch_num_pages;
        }

        slice_cursor.advance();

        if (do_reduce && (i == (ring_size - 1))) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
            sem_target = 0;
        }
    }
}

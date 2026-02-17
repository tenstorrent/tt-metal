// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>
#include <utility>
#include "api/debug/dprint.h"
#include "strided_ring_reduce_scatter_common.hpp"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

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
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(10);
constexpr uint32_t slice_C = get_compile_time_arg_val(11);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(12);
constexpr uint32_t dim = get_compile_time_arg_val(13);
constexpr uint32_t mm_M_unit_blocks_per_core = get_compile_time_arg_val(14);
constexpr uint32_t mm_N_full_blocks_per_slice = get_compile_time_arg_val(15);
constexpr uint32_t mm_block_ht = get_compile_time_arg_val(16);
constexpr uint32_t mm_cores_y = get_compile_time_arg_val(17);
constexpr uint32_t mm_N_full_block_wt = get_compile_time_arg_val(18);
constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(19);
constexpr uint32_t chunks_per_mm_N_full_block = get_compile_time_arg_val(20);

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
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 21;

#ifdef INPUT_IS_SHARDED
    constexpr uint32_t ct_offset = 7;

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
    arg_idx += input_rt_increment;
#else
    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
    auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);
#endif

#ifdef INTERMEDIATE_IS_SHARDED
    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx + ct_offset),       // Memory layout
        get_compile_time_arg_val(ct_idx + ct_offset + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + ct_offset + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + ct_offset + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(ct_idx + ct_offset + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(ct_idx + ct_offset + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + ct_offset + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    arg_idx += intermediate_rt_increment;
#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto intermediate_tensor_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address, page_size);
#endif
    /**
    Note that each minimal matmul core outputs a "big" block of size mm_M_block_ht x mm_N_full_block_wt.
    These are produced in mm_block_ht x mm_block_wt "unit" blocks in a block-row-major order.
    There are mm_N_full_blocks_per_slice big blocks in a single slice along the width dimension
    (mm_N_full_blocks_per_slice = slice_Wt / mm_N_full_block_wt).
    There are mm_M_unit_blocks_per_core unit blocks per minimal matmul core in the y-direction.
    (mm_M_unit_blocks_per_core = slice_Ht / mm_block_ht / mm_cores_y).

    TODO: change to matmul units
    A chunk is defined as a strided block row of tiles produced by the matmul cores within a given number of steps.
    More precisely, if chunk_width_in_tiles = t, then the first chunk contains
    all tiles corresponding to this slice and produced by the matmul cores exactly after they produce t tiles each,
    the second chunk contains subsequent tiles produced by the matmul cores after they produce 2t tiles each, etc.

    The algorithm iterates over the chunks and reduces each chunk with the ring reduce scatter algorithm.
    A full reduction is performed so that before going to the next chunk, all tiles in the current chunk
    are reduced, i.e., each device stores the final reduced value of the given chunk.
    (From the perspective of the reader kernel kernel, the chunk is fully taken care of.)
    Note that each chunk can be composed of multiple pieces if mm_N_full_blocks_per_slice > 1.
    */
    ASSERT(dim == 3);
    ASSERT(slice_C == 1);

    const uint32_t batch_size = input_tensor_B;
    const uint32_t last_mm_core_idx = mm_cores_y - 1;
    const uint32_t tiles_ht_per_core = mm_block_ht * mm_M_unit_blocks_per_core;
    const uint32_t effective_worker_id = worker_id + (direction ? num_workers : 0);
    const uint32_t effective_advance_by_tiles = 2 * num_workers;

    uint32_t out_ready_sem_target = 0;

    for (uint32_t b = 0; b < batch_size; b++) {
        const uint32_t batch_offset = input_batch_num_pages * b;

        for (uint32_t m_block_iter = 0; m_block_iter < mm_M_unit_blocks_per_core; m_block_iter++) {
            for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_full_block; chunk_idx++) {
                const uint32_t effective_chunk_width_in_tiles =
                    get_effective_chunk_width_in_tiles(chunk_idx, chunk_width_in_tiles, mm_N_full_block_wt);
                const uint32_t effective_subchunk_size = mm_block_ht * effective_chunk_width_in_tiles;
                int32_t slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

                for (uint32_t i = 0; i < ring_size; i++) {
                    const bool do_reduce = i != 0;
                    const uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;
                    const uint32_t actual_slice_idx = wrap_slice_idx(slice_idx, direction, ring_size);

                    // Wait for all chunk_piece_idx tiles for this ring iteration to be written
                    if (do_reduce) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), out_ready_sem_target + 1);
                        out_ready_sem_target++;
                    }

                    for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_full_blocks_per_slice;
                         chunk_piece_idx++) {
                        uint32_t first_tile_row_in_mm_M_block = 0;
                        uint32_t first_chunk_col_in_tiles = 0;
                        uint32_t first_mm_core_idx = 0;
                        // get the first tile coordinates for the current chunk piece
                        get_next_tile_coordinates(
                            first_tile_row_in_mm_M_block,
                            first_chunk_col_in_tiles,
                            first_mm_core_idx,
                            effective_worker_id,
                            effective_subchunk_size,
                            effective_chunk_width_in_tiles,
                            mm_block_ht);
                        uint32_t tiles_to_read = how_many_tiles_to_read_formula(
                            first_tile_row_in_mm_M_block,
                            first_chunk_col_in_tiles,
                            first_mm_core_idx,
                            effective_advance_by_tiles,
                            last_mm_core_idx,
                            effective_subchunk_size,
                            effective_chunk_width_in_tiles);

                        while (tiles_to_read > 0) {
                            const uint32_t tiles_to_read_in_this_step = std::min(tiles_to_read, tile_granularity);
                            tiles_to_read -= tiles_to_read_in_this_step;

                            cb_reserve_back(cb_in0, tile_granularity);
                            uint32_t l1_write_addr = get_write_ptr(cb_in0);
                            uint32_t intermediate_l1_write_addr;
                            if (do_reduce) {
                                cb_reserve_back(cb_intermediate_id, tile_granularity);
                                intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                            }

                            for (uint32_t j = 0; j < tiles_to_read_in_this_step; ++j) {
                                auto [slice_tile_idx, global_tile_idx] = coordinates_to_tile_indices(
                                    first_tile_row_in_mm_M_block,
                                    first_chunk_col_in_tiles,
                                    first_mm_core_idx,
                                    chunk_piece_idx,
                                    m_block_iter,
                                    chunk_idx,
                                    mm_N_full_block_wt,
                                    tiles_ht_per_core,
                                    mm_block_ht,
                                    chunk_width_in_tiles,
                                    actual_slice_idx,
                                    slice_Wt,
                                    input_tensor_Wt);
                                const uint32_t input_tile_id = global_tile_idx + batch_offset;

                                const uint64_t noc_read_addr = get_noc_addr(input_tile_id, input_tensor_addrgen);
                                noc_async_read(noc_read_addr, l1_write_addr, page_size);
                                l1_write_addr += page_size;
                                if (do_reduce) {
                                    uint64_t intermediate_noc_read_addr =
                                        get_noc_addr(global_tile_idx, intermediate_tensor_addrgen);
                                    noc_async_read(intermediate_noc_read_addr, intermediate_l1_write_addr, page_size);
                                    intermediate_l1_write_addr += page_size;
                                }

                                get_next_tile_coordinates(
                                    first_tile_row_in_mm_M_block,
                                    first_chunk_col_in_tiles,
                                    first_mm_core_idx,
                                    effective_advance_by_tiles,
                                    effective_subchunk_size,
                                    effective_chunk_width_in_tiles,
                                    mm_block_ht);
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_in0, tile_granularity);
                            if (do_reduce) {
                                cb_push_back(cb_intermediate_id, tile_granularity);
                            }
                        }
                    }

                    // Next slice idx
                    if (direction) {
                        slice_idx--;
                    } else {
                        slice_idx++;
                    }
                }
            }
        }
        // Reset the semaphore before the next batch
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        out_ready_sem_target = 0;
    }
}

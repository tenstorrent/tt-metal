// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>
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
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(10);
constexpr uint32_t slice_C = get_compile_time_arg_val(11);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(12);
constexpr uint32_t dim = get_compile_time_arg_val(13);
constexpr uint32_t mm_M_unit_blocks_per_core = get_compile_time_arg_val(14);
constexpr uint32_t mm_block_ht = get_compile_time_arg_val(15);
constexpr uint32_t mm_cores_y = get_compile_time_arg_val(16);
constexpr uint32_t mm_N_full_block_wt = get_compile_time_arg_val(17);
constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(18);
constexpr uint32_t chunks_per_mm_N_full_block = get_compile_time_arg_val(19);
constexpr uint32_t mm_block_wt = get_compile_time_arg_val(20);
constexpr uint32_t slice_Ht_per_core = get_compile_time_arg_val(21);
// [22]=fuse_mm_op (via FUSE_MM_OP_SIGNALER define)
constexpr uint32_t slice_Ht = get_compile_time_arg_val(23);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx = 24;  // [20]=mm_block_wt, [21]=slice_Ht_per_core, [22]=fuse_mm_op (via
                                     // FUSE_MM_OP_SIGNALER define), [23]=slice_Ht

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
#ifdef FUSE_MM_OP_SIGNALER
    size_t mm_op_ready_sem = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t mm_sem_target = 0;
#endif
    /**
    Iterate over chunks in the row-major order and reduce-scatter each chunk, one by one.
    In particular, for each chunk, perform a full ring reduce-scatter iteration before going to the next one.
    Note that each chunk can be composed of multiple pieces if mm_N_full_blocks_per_slice > 1.
    */
    ASSERT(dim == 3);      // strided reduce-scatter only supports scattering on dim 3
    ASSERT(slice_C == 1);  // note: this can be relaxed if needed but is omitted to avoid nested loops

    const uint32_t batch_size = input_tensor_B;
    const uint32_t last_mm_core_idx = mm_cores_y - 1;
    // Use actual row count per core (not padded), so coordinates_to_slice_coordinates
    // produces correct absolute row offsets across MM cores.
    const uint32_t tiles_ht_per_core = slice_Ht_per_core;

    // Each worker handles every 'effective_advance_by_tiles'-th tile, starting from offset 'effective_worker_id'.
    const uint32_t effective_worker_id = worker_id + (direction ? num_workers : 0);
    const uint32_t effective_advance_by_tiles = 2 * num_workers;

    // Snapshot the semaphore's value at startup
    uint32_t out_ready_sem_target = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem);

    for (uint32_t b = 0; b < batch_size; b++) {
        const uint32_t batch_offset = input_batch_num_pages * b;

        for (uint32_t m_block_iter = 0; m_block_iter < mm_M_unit_blocks_per_core; m_block_iter++) {
            const uint32_t current_mm_block_ht =
                get_current_mm_block_ht(m_block_iter, mm_M_unit_blocks_per_core, mm_block_ht, slice_Ht_per_core);
            for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_full_block; chunk_idx++) {
                const uint32_t effective_chunk_width_in_tiles =
                    get_effective_chunk_width_in_tiles(chunk_idx, chunk_width_in_tiles, mm_N_full_block_wt);
                const uint32_t effective_subchunk_size = current_mm_block_ht * effective_chunk_width_in_tiles;
                int32_t slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

#ifdef FUSE_MM_OP_SIGNALER
                // Wait for matmul to finish writing the output blocks for this chunk.
                // The matmul signals in a strided pattern: value k means k mm_blocks have been
                // written in EACH N-full-block, so ceil(effective_chunk_width / mm_block_wt)
                // signals guarantees all N-full-blocks covering this chunk are ready.
                const uint32_t sem_increment = (effective_chunk_width_in_tiles + mm_block_wt - 1) / mm_block_wt;
                mm_sem_target += sem_increment;
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mm_op_ready_sem), mm_sem_target);
#endif
                // Run a full ring reduce-scatter for the current chunk.
                for (uint32_t i = 0; i < ring_size; i++) {
                    const bool do_reduce = i != 0;
                    const uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;
                    const uint32_t actual_slice_idx = wrap_slice_idx(slice_idx, direction, ring_size);

                    const auto [mm_N_full_blocks_per_slice, cols_before_actual_slice] =
                        get_slice_N_block_info(actual_slice_idx, slice_Wt, mm_N_full_block_wt);
                    // Wait for all chunk_piece_idx tiles for this ring iteration to be written by the neighboring
                    // device
                    if (do_reduce) {
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), out_ready_sem_target + 1);
                        out_ready_sem_target++;
                    }

                    for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_full_blocks_per_slice;
                         chunk_piece_idx++) {
                        uint32_t tile_row_in_mm_M_unit_block = 0;
                        uint32_t chunk_col_in_tiles = 0;
                        uint32_t mm_core_idx = 0;
                        // Get the first tile coordinates for the current chunk piece
                        get_next_tile_coordinates(
                            tile_row_in_mm_M_unit_block,
                            chunk_col_in_tiles,
                            mm_core_idx,
                            effective_worker_id,
                            effective_subchunk_size,
                            effective_chunk_width_in_tiles,
                            current_mm_block_ht);
                        uint32_t tiles_to_read = how_many_tiles_to_read_formula(
                            tile_row_in_mm_M_unit_block,
                            chunk_col_in_tiles,
                            mm_core_idx,
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
                                const auto [slice_row, slice_col] = coordinates_to_slice_coordinates(
                                    tile_row_in_mm_M_unit_block,
                                    chunk_col_in_tiles,
                                    mm_core_idx,
                                    chunk_piece_idx,
                                    m_block_iter,
                                    chunk_idx,
                                    mm_N_full_block_wt,
                                    tiles_ht_per_core,
                                    mm_block_ht,
                                    chunk_width_in_tiles);

                                if (slice_row < slice_Ht && slice_col >= cols_before_actual_slice &&
                                    slice_col < cols_before_actual_slice + slice_Wt) {
                                    const uint32_t global_tile_idx = slice_coordinates_to_global_tile_index(
                                        slice_row,
                                        slice_col - cols_before_actual_slice,
                                        actual_slice_idx,
                                        slice_Wt,
                                        input_tensor_Wt);
                                    const uint32_t input_tile_id = global_tile_idx + batch_offset;
                                    noc_async_read(
                                        get_noc_addr(input_tile_id, input_tensor_addrgen), l1_write_addr, page_size);
                                    if (do_reduce) {
                                        noc_async_read(
                                            get_noc_addr(global_tile_idx, intermediate_tensor_addrgen),
                                            intermediate_l1_write_addr,
                                            page_size);
                                    }
                                }
                                // Always advance: CB position i corresponds to iteration tile i,
                                // so the writer can find valid tile data at the correct packet slot.
                                l1_write_addr += page_size;
                                if (do_reduce) {
                                    intermediate_l1_write_addr += page_size;
                                }

                                get_next_tile_coordinates(
                                    tile_row_in_mm_M_unit_block,
                                    chunk_col_in_tiles,
                                    mm_core_idx,
                                    effective_advance_by_tiles,
                                    effective_subchunk_size,
                                    effective_chunk_width_in_tiles,
                                    current_mm_block_ht);
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_in0, tile_granularity);
                            if (do_reduce) {
                                cb_push_back(cb_intermediate_id, tile_granularity);
                            }
                        }
                    }
                    // Move to the next slice
                    slice_idx += direction ? -1 : 1;
                }
            }
        }
        // Reset between batches so the counter doesn't overflow across batches.
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        out_ready_sem_target = 0;

#ifdef FUSE_MM_OP_SIGNALER
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mm_op_ready_sem), 0);
        mm_sem_target = 0;
#endif
    }

    // Explicit cleanup: guarantee the semaphore is 0 when this kernel exits
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_chunk_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"

// Drains one query brick's normalized output per work item.
//
// Output stays in BRICKED order, matching the input, so the next block in the stage consumes
// it directly -- the permute happens once at stage entry and once at exit, not per block.
// Nothing here knows about context windows; it writes the tile row it is handed.

namespace kernel_args = ttnn::transformer::neighborhood::kernel_args;
namespace layout = ttnn::transformer::neighborhood::chunk_layout;

#ifndef NA_PATH_KIND
#define NA_PATH_KIND 0
#endif
#ifndef NA_SKIP_IF
#if NA_PATH_KIND == 1
#define NA_SKIP_IF 2u
#elif NA_PATH_KIND == 2
#define NA_SKIP_IF 3u
#else
#define NA_SKIP_IF 0u
#endif
#endif

#ifndef NA_HAS_PATH_SKIP
__attribute__((noinline, noclone)) bool na_path_skips_chunk(uint32_t) { return false; }
#endif

#ifdef NA_SKIP_IF
template <uint32_t SkipIf>
__attribute__((noinline, noclone)) bool na_write_kind(uint32_t packed_width) {
    return (2u + (packed_width >> 31)) != SkipIf;
}
#endif

__attribute__((noinline, noclone)) bool na_should_write(uint32_t packed_width, uint32_t skip_if) {
    return (2u + (packed_width >> 31)) != skip_if;
}

void kernel_main() {
    constexpr uint32_t head_count = get_compile_time_arg_val(kernel_args::writer_arg::head_count);
    constexpr uint32_t brick_count = get_compile_time_arg_val(kernel_args::writer_arg::brick_count);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(kernel_args::writer_arg::head_dim_tiles);
    constexpr uint32_t bricks_per_query_chunk =
        get_compile_time_arg_val(kernel_args::writer_arg::bricks_per_query_chunk);
    constexpr kernel_args::AxisExtents query_chunk_bricks{
        get_compile_time_arg_val(kernel_args::writer_arg::query_chunk_bricks_time),
        get_compile_time_arg_val(kernel_args::writer_arg::query_chunk_bricks_height),
        get_compile_time_arg_val(kernel_args::writer_arg::query_chunk_bricks_width)};
    constexpr kernel_args::AxisExtents volume_chunks{
        get_compile_time_arg_val(kernel_args::writer_arg::volume_chunks_time),
        get_compile_time_arg_val(kernel_args::writer_arg::volume_chunks_height),
        get_compile_time_arg_val(kernel_args::writer_arg::volume_chunks_width)};
    constexpr kernel_args::AxisExtents volume_bricks{
        get_compile_time_arg_val(kernel_args::writer_arg::volume_bricks_time),
        get_compile_time_arg_val(kernel_args::writer_arg::volume_bricks_height),
        get_compile_time_arg_val(kernel_args::writer_arg::volume_bricks_width)};
    constexpr uint32_t chunk_count = volume_chunks.time * volume_chunks.height * volume_chunks.width;

    constexpr uint32_t path_mode = get_compile_time_arg_val(kernel_args::writer_arg::path_mode);
    constexpr uint32_t skip_unowned = get_compile_time_arg_val(kernel_args::writer_arg::skip_unowned);
    constexpr uint32_t skip_if_bit = get_compile_time_arg_val(kernel_args::writer_arg::skip_if_bit);
    (void)path_mode;

    constexpr auto output_accessor_args = TensorAccessorArgs<kernel_args::writer_arg::COUNT>();
    constexpr auto origin_accessor_args = TensorAccessorArgs<output_accessor_args.next_compile_time_args_offset()>();

    uint32_t argument_index = 0;
    const uint32_t output_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t gather_origin_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t work_item_start = get_arg_val<uint32_t>(argument_index++);
    const uint32_t work_item_count = get_arg_val<uint32_t>(argument_index++);
    const uint32_t tile_and_skip = get_arg_val<uint32_t>(argument_index++);
    const uint32_t tile_bytes = tile_and_skip & 0xffffu;
    const uint32_t skip_if_runtime = tile_and_skip >> 16;

    const auto output_writer = TensorAccessor(output_accessor_args, output_address);
    CircularBuffer cb_output(kernel_args::cb_output);
    Noc noc;
#ifndef NA_SKIP_IF_BIT
    if (skip_unowned == 0) {
        (void)gather_origin_address;
        (void)sizeof(origin_accessor_args);
        (void)skip_if_bit;
    }
#endif

    // The reduce identity, built once and left resident: the compute kernel's row max and row
    // sum both reduce against it. Generated here rather than uploaded because it is one tile
    // of a known constant, and the writer is otherwise idle until the first output appears.
    // bfloat16 1.0 packed into both halves of a uint32, the form the helper expects.
    constexpr uint32_t reduce_identity_bits = 0x3F803F80;
    wh_generate_reduce_scaler(kernel_args::cb_reduce_scalar, reduce_identity_bits);

    // A genuine zero tile. matmul_blocks folds the mask in as `dst += zero + mask`, so if this
    // is not actually zero the scores are quietly corrupted rather than obviously broken.
    CircularBuffer cb_zero(kernel_args::cb_zero);
    cb_zero.reserve_back(1);
    volatile tt_l1_ptr uint32_t* zero_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_zero.get_write_ptr());
    for (uint32_t word = 0; word < (32 * 32 * sizeof(uint16_t)) / sizeof(uint32_t); ++word) {
        zero_tile[word] = 0;
    }
    cb_zero.push_back(1);

    // Ones down column 0. sub_exp only partially reduces each chunk's sum, so the final
    // within-tile row reduction is deferred out of the KV loop and done as a matmul here.
    generate_bcast_col_scalar(CircularBuffer(kernel_args::cb_column_identity), reduce_identity_bits);

    for (uint32_t work_item = work_item_start; work_item < work_item_start + work_item_count; ++work_item) {
        // Same decomposition as the reader: one work item is one query chunk.
        const uint32_t chunk_index = work_item % chunk_count;
        const uint32_t head_index = (work_item / chunk_count) % head_count;
        const uint32_t batch_index = work_item / (chunk_count * head_count);

        const layout::BrickCoordinate chunk_origin =
            layout::chunk_origin_brick(chunk_index, volume_chunks, query_chunk_bricks);

        bool write_this_chunk = true;
#ifdef NA_HAS_PATH_SKIP
        {
            const auto origin_reader = TensorAccessor(origin_accessor_args, gather_origin_address);
            CircularBuffer cb_writer_origin(kernel_args::cb_writer_origin);
            cb_writer_origin.reserve_back(1);
            const uint32_t origin_write_pointer = cb_writer_origin.get_write_ptr();
            noc.async_read(
                origin_reader,
                CoreLocalMem<uint32_t>(origin_write_pointer),
                kernel_args::GATHER_ORIGIN_ROW_BYTES,
                {.page_id = chunk_index},
                {});
            noc.async_read_barrier();
            CoreLocalMem<volatile uint32_t> origin_mem(origin_write_pointer);
            volatile uint32_t edge_token = origin_mem[kernel_args::gather_origin_column::skip_edge_token];
            if constexpr (path_mode != 2) {
                if (edge_token == 0xFFFFFFFFu) {
                    write_this_chunk = false;
                }
            } else {
                if (edge_token != 0xFFFFFFFFu) {
                    write_this_chunk = false;
                }
            }
            cb_writer_origin.push_back(1);
            cb_writer_origin.pop_front(1);
        }
#else
        (void)skip_unowned;
        (void)skip_if_bit;
        (void)skip_if_runtime;
#endif

        cb_output.wait_front(head_dim_tiles * bricks_per_query_chunk);
        uint32_t read_pointer = cb_output.get_read_ptr();
        for (uint32_t brick_in_chunk = 0; brick_in_chunk < bricks_per_query_chunk; ++brick_in_chunk) {
            const layout::BrickCoordinate brick =
                layout::brick_within_chunk(brick_in_chunk, chunk_origin, query_chunk_bricks);
            // A chunk may hang off the end of the volume; those bricks have no home to write to.
            if (!write_this_chunk || !layout::brick_is_inside(brick, volume_bricks)) {
                read_pointer += head_dim_tiles * tile_bytes;
                continue;
            }
            const uint32_t first_tile = layout::tile_offset(
                batch_index,
                layout::brick_index(brick, volume_bricks),
                head_index,
                brick_count,
                head_count,
                head_dim_tiles);
            for (uint32_t head_dim_tile = 0; head_dim_tile < head_dim_tiles; ++head_dim_tile) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(read_pointer),
                    output_writer,
                    tile_bytes,
                    {},
                    {.page_id = first_tile + head_dim_tile});
                read_pointer += tile_bytes;
            }
        }
        noc.async_write_barrier();
        cb_output.pop_front(head_dim_tiles * bricks_per_query_chunk);
    }
}

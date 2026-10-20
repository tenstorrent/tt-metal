// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows feed for the combine (data movement only), on the feeder cores (one per ring position). Feeder r copies
// its W2 output columns [first_tile, first_tile + tiles) of every routed row from the row buffer into moe_compute's
// double buffer on the combine cores, one local expert at a time, with the handshake of the ring program's dm1
// (dm1.cpp): the columns may span several width shards; before expert e the feeder waits until the height_shards
// combine cores of each of those width shards released half e % 2 (one increment each per expert), after expert e it
// increments each of those cores' semaphore once. Expert e's rows go to the height shards in order (earlier shards
// take the remainder), row i of a height shard at (e % 2) * half_bytes + i * shard_row_bytes; the columns land at
// their offset within each width shard.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {

constexpr uint32_t local_experts = get_named_compile_time_arg_val("local_experts");
constexpr uint32_t height_shards = get_named_compile_time_arg_val("height_shards");
constexpr uint32_t width_shards = get_named_compile_time_arg_val("width_shards");
constexpr uint32_t shard_width_tiles = get_named_compile_time_arg_val("shard_width_tiles");
constexpr uint32_t tile_width_bytes = get_named_compile_time_arg_val("tile_width_bytes");
constexpr uint32_t half_bytes = get_named_compile_time_arg_val("half_bytes");
constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
constexpr uint32_t batch_rows = get_named_compile_time_arg_val("batch_rows");
constexpr uint32_t slice_stride = get_named_compile_time_arg_val("slice_stride");
// routing table block (rows_reader.cpp): byte offsets of the per-expert counts, first rows and rows; its size
constexpr uint32_t table_counts = get_named_compile_time_arg_val("table_counts");
constexpr uint32_t table_offsets = get_named_compile_time_arg_val("table_offsets");
constexpr uint32_t table_rows = get_named_compile_time_arg_val("table_rows");
constexpr uint32_t table_bytes = get_named_compile_time_arg_val("table_bytes");
constexpr uint32_t scratch_stage = get_named_compile_time_arg_val("s_stage");  // two staging buffers of batch_rows
constexpr uint32_t combine_sync_semaphore_id = get_named_compile_time_arg_val("combine_sync_semaphore_id");
constexpr uint32_t shard_row_bytes = shard_width_tiles * tile_width_bytes;

// The feeder's output columns [first_tile, first_tile + tiles) in at most three width-shard pieces: offset in the
// staged slice, width shard, offset in that shard's row, bytes.
struct SlicePieces {
    uint32_t slice_offset[3], shard[3], shard_offset[3], bytes[3];
    uint32_t count = 0;
    FORCE_INLINE SlicePieces(uint32_t first_tile, uint32_t tiles) {
        for (uint32_t tile = first_tile, done = 0; done < tiles && count < 3; ++count) {
            const uint32_t in_shard = tile % shard_width_tiles;
            const uint32_t n =
                shard_width_tiles - in_shard < tiles - done ? shard_width_tiles - in_shard : tiles - done;
            slice_offset[count] = done * tile_width_bytes;
            shard[count] = tile / shard_width_tiles;
            shard_offset[count] = in_shard * tile_width_bytes;
            bytes[count] = n * tile_width_bytes;
            tile += n;
            done += n;
        }
    }
};

// Combine core of shard height_shard * width_shards + width_shard: its virtual (x, y) in the runtime args from
// `combine_cores_arg`.
FORCE_INLINE uint32_t combine_core_x(uint32_t combine_cores_arg, uint32_t shard) {
    return get_arg_val<uint32_t>(combine_cores_arg + 2 * shard);
}
FORCE_INLINE uint32_t combine_core_y(uint32_t combine_cores_arg, uint32_t shard) {
    return get_arg_val<uint32_t>(combine_cores_arg + 2 * shard + 1);
}

FORCE_INLINE uint32_t staging_buffer(uint32_t scratch, uint32_t batch) {
    return scratch + scratch_stage + (batch & 1) * batch_rows * slice_stride;
}

FORCE_INLINE uint32_t rows_in_batch(uint32_t expert_rows, uint32_t batch) {
    const uint32_t first = batch * batch_rows;
    return expert_rows - first < batch_rows ? expert_rows - first : batch_rows;
}

// The feeder's columns of the expert's rows of batch `batch` into staging buffer batch % 2.
template <typename RowsAccessor>
FORCE_INLINE void read_batch(
    const Noc& noc,
    const RowsAccessor& rows,
    uint32_t scratch,
    uint32_t expert_first_row,
    uint32_t expert_rows,
    uint32_t batch,
    uint32_t first_tile,
    uint32_t tiles) {
    const uint32_t first = batch * batch_rows;
    const uint32_t n = rows_in_batch(expert_rows, batch);
    const uint32_t buffer = staging_buffer(scratch, batch);
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(
            rows,
            CoreLocalMem<uint32_t>(buffer + i * slice_stride),
            tiles * tile_width_bytes,
            {.page_id = expert_first_row + first + i, .offset_bytes = first_tile * tile_width_bytes},
            {});
    }
}

// The expert's rows over the height shards in order; earlier shards take the remainder.
struct HeightShardCursor {
    uint32_t base_rows, remainder;
    uint32_t shard = 0, row = 0, shard_rows;
    explicit HeightShardCursor(uint32_t expert_rows) :
        base_rows(expert_rows / height_shards),
        remainder(expert_rows % height_shards),
        shard_rows(base_rows + (remainder ? 1 : 0)) {}
    FORCE_INLINE void advance() {
        if (++row == shard_rows) {
            ++shard;
            row = 0;
            shard_rows = base_rows + (shard < remainder ? 1 : 0);
        }
    }
};

// One staged batch to the combine cores' double-buffer half: each row's pieces to the shard the cursor is on.
FORCE_INLINE void write_batch(
    uint32_t staged,
    uint32_t n,
    const SlicePieces& pieces,
    uint32_t combine_cores_arg,
    uint32_t double_buffer,
    uint32_t half,
    HeightShardCursor& cursor) {
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t p = 0; p < pieces.count; ++p) {
            const uint32_t shard = cursor.shard * width_shards + pieces.shard[p];
            const uint32_t x = combine_core_x(combine_cores_arg, shard);
            const uint32_t y = combine_core_y(combine_cores_arg, shard);
            // dataflow_api write: Noc::async_write in this loop makes the program 5-6 % slower
            noc_async_write(
                staged + i * slice_stride + pieces.slice_offset[p],
                get_noc_addr(x, y, double_buffer + half + cursor.row * shard_row_bytes + pieces.shard_offset[p]),
                pieces.bytes[p]);
        }
        cursor.advance();
    }
}

// Once per expert: one increment to every combine core of the feeder's width shards.
FORCE_INLINE void signal_combine_cores(
    const Noc& noc, Semaphore<>& semaphore, const SlicePieces& pieces, uint32_t combine_cores_arg) {
    for (uint32_t p = 0; p < pieces.count; ++p) {
        for (uint32_t h = 0; h < height_shards; ++h) {
            const uint32_t shard = h * width_shards + pieces.shard[p];
            semaphore.up(noc, combine_core_x(combine_cores_arg, shard), combine_core_y(combine_cores_arg, shard), 1);
        }
    }
}

}  // namespace

void kernel_main() {
    CircularBuffer cb_scratch(get_named_compile_time_arg_val("cb_scratch"));
    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto table_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t rows_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t double_buffer = get_arg_val<uint32_t>(arg++);  // same L1 address on every combine core
    const uint32_t first_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t combine_cores_arg = arg;

    const Noc noc;
    cb_scratch.reserve_back(1);
    const uint32_t scratch = cb_scratch.get_write_ptr();
    const auto table = TensorAccessor(table_args, table_addr, table_bytes);
    noc.async_read(table, CoreLocalMem<uint32_t>(scratch), table_rows, {.page_id = 0}, {});
    noc.async_read_barrier();
    const tt_l1_ptr uint16_t* expert_row_counts = reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + table_counts);
    const tt_l1_ptr uint16_t* expert_first_rows = reinterpret_cast<const tt_l1_ptr uint16_t*>(scratch + table_offsets);

    const SlicePieces pieces(first_tile, tiles);
    const auto rows = TensorAccessor(rows_args, rows_addr, row_bytes);
    Semaphore<> semaphore(combine_sync_semaphore_id);
    uint32_t released = 0;
    for (uint32_t e = 0; e < local_experts; ++e) {
        const uint32_t n = expert_row_counts[e], first_row = expert_first_rows[e];
        const uint32_t half = (e & 1) * half_bytes;
        const uint32_t batches = (n + batch_rows - 1) / batch_rows;
        if (batches) {
            read_batch(noc, rows, scratch, first_row, n, 0, first_tile, tiles);
        }
        semaphore.wait_min(released);  // the combine released this half
        HeightShardCursor cursor(n);
        for (uint32_t batch = 0; batch < batches; ++batch) {
            noc.async_read_barrier();
            noc.async_writes_flushed();  // the previous batch left the other staging buffer
            if (batch + 1 < batches) {
                read_batch(noc, rows, scratch, first_row, n, batch + 1, first_tile, tiles);
            }
            write_batch(
                staging_buffer(scratch, batch),
                rows_in_batch(n, batch),
                pieces,
                combine_cores_arg,
                double_buffer,
                half,
                cursor);
        }
        noc.async_write_barrier();  // the rows are in the combine cores' L1 before their semaphore moves
        signal_combine_cores(noc, semaphore, pieces, combine_cores_arg);
        released += height_shards * pieces.count;  // every combine core signalled releases this feeder once per expert
    }
    semaphore.wait_min(released);  // the combine's last release, then reset for the next call
    semaphore.set(0);
    noc.async_atomic_barrier();
}

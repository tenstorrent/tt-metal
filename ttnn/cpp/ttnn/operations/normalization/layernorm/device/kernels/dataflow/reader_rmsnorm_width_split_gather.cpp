// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the width-split RMSNorm: streams this core's input block in, then runs the cross-core
// reduce.
//
// Every core's compute kernel leaves, for each tile row, its partial sum of squares as a row
// vector: row 0 of faces 0 and 1 of a tile. Each core writes those two face rows into its own row
// of the root's gather tiles (32 cores to a tile) and bumps the root's gather semaphore. The root
// then reduces the gather tiles column-wise into 1/sqrt(mean + eps), and multicasts that back with
// the broadcast semaphore. Pushing two face rows per tile row from each core avoids both a global
// readiness barrier and reading whole partial tiles back from every core.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"
#include "interleaved_block_io.hpp"

void kernel_main() {
    constexpr auto num_tile_rows = get_arg(args::num_tile_rows);
    constexpr auto gather_tiles_per_row = get_arg(args::gather_tiles_per_row);
    constexpr auto num_cores = get_arg(args::num_cores);
    constexpr auto face_row_bytes = get_arg(args::face_row_bytes);
    constexpr auto face_bytes = get_arg(args::face_bytes);
    constexpr uint32_t gather_tiles = num_tile_rows * gather_tiles_per_row;
    constexpr uint32_t rows_per_face = 16;
    constexpr uint32_t rows_per_tile = 32;

    const bool is_root = get_arg(args::is_root) == 1;
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t root_noc_x = get_arg(args::root_noc_x);
    const uint32_t root_noc_y = get_arg(args::root_noc_y);
    const uint32_t mcast_start_x = get_arg(args::mcast_start_x);
    const uint32_t mcast_start_y = get_arg(args::mcast_start_y);
    const uint32_t mcast_end_x = get_arg(args::mcast_end_x);
    const uint32_t mcast_end_y = get_arg(args::mcast_end_y);

    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto Kt = get_arg(args::Kt);
    const uint32_t width_tile_start = get_arg(args::width_tile_start);
    const uint32_t num_valid_tiles = get_arg(args::num_valid_tiles);

    Noc noc;
    Semaphore<> gather_sem(sem::gather_sem);
    Semaphore<> bcast_sem(sem::bcast_sem);
    const UnicastEndpoint remote_ep;
    MulticastEndpoint mcast_ep;
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_partial_t(dfb::partial_t);
    DataflowBuffer dfb_gather(dfb::gather);
    DataflowBuffer dfb_rstd(dfb::rstd);
    DataflowBuffer dfb_global(dfb::global);

    // This kernel runs on the NoC preferred for DRAM reads, so it streams the input in before it
    // turns to the exchange.
    issue_block_reads(
        noc, TensorAccessor(tensor::src), dfb_in, num_tile_rows, block_w, Kt, width_tile_start, num_valid_tiles);

    const uint32_t tile_bytes = dfb_gather.get_tile_size();
    // Every core carries the same buffers at the same addresses, so this core's gather and global
    // addresses are also the root's and every other core's.
    dfb_gather.reserve_back(gather_tiles);
    const uint32_t gather_base = dfb_gather.get_write_ptr();
    dfb_global.reserve_back(num_tile_rows);
    const uint32_t global_base = dfb_global.get_write_ptr();

    auto row_offset = [](uint32_t row) {
        return (row / rows_per_face) * 2 * face_bytes + (row % rows_per_face) * face_row_bytes;
    };

    // Rows past the last core are never written; the root zeroes them so the column reduce ignores
    // them. No other core touches these rows, so this cannot race the gather writes. They sit at the
    // end of the last gather tile of each tile row: the tail rows of the face pair they start in,
    // then, if that is the top pair, the whole bottom pair.
    constexpr uint32_t first_unused_row = num_cores % rows_per_tile;
    if (is_root && first_unused_row != 0) {
        constexpr uint32_t pair_bytes = 2 * face_bytes;
        constexpr uint32_t tail_rows = rows_per_face - first_unused_row % rows_per_face;
        for (uint32_t r = 0; r < num_tile_rows; ++r) {
            const uint32_t tile_offset = (r * gather_tiles_per_row + gather_tiles_per_row - 1) * tile_bytes;
            const uint32_t tail_offset = tile_offset + row_offset(first_unused_row);
            noc.async_write_zeros(dfb_gather, tail_rows * face_row_bytes, {.offset_bytes = tail_offset});
            noc.async_write_zeros(dfb_gather, tail_rows * face_row_bytes, {.offset_bytes = tail_offset + face_bytes});
            if constexpr (first_unused_row < rows_per_face) {
                noc.async_write_zeros(dfb_gather, pair_bytes, {.offset_bytes = tile_offset + pair_bytes});
            }
        }
        noc.write_zeros_l1_barrier();
    }
    noc.async_read_barrier();
    dfb_in.push_back(static_cast<uint16_t>(num_tile_rows * block_w));

    dfb_partial_t.wait_front(num_tile_rows);
    const uint32_t slot_offset = (core_index / rows_per_tile) * tile_bytes + row_offset(core_index % rows_per_tile);
    for (uint32_t r = 0; r < num_tile_rows; ++r) {
        const uint32_t dst = gather_base + r * gather_tiles_per_row * tile_bytes + slot_offset;
        for (uint32_t half = 0; half < 2; ++half) {
            noc.async_write(
                dfb_partial_t,
                remote_ep,
                face_row_bytes,
                {.offset_bytes = r * tile_bytes + half * face_bytes},
                {.noc_x = root_noc_x, .noc_y = root_noc_y, .addr = dst + half * face_bytes});
        }
    }
    noc.async_write_barrier();
    dfb_partial_t.pop_front(num_tile_rows);

    if (!is_root) {
        gather_sem.up(noc, root_noc_x, root_noc_y, 1);
        bcast_sem.wait(1);
        bcast_sem.set(0);
        dfb_global.push_back(num_tile_rows);
        // The increment is tracked apart from writes; drain it before the kernel ends.
        noc.async_atomic_barrier();
        return;
    }

    gather_sem.wait(num_cores - 1);
    gather_sem.set(0);
    dfb_gather.push_back(gather_tiles);

    dfb_rstd.wait_front(num_tile_rows);
    const uint32_t rstd_bytes = num_tile_rows * tile_bytes;
    // set_multicast sends this semaphore's own value, linked behind the data so it arrives after it.
    bcast_sem.set(1);
    noc.async_write_multicast(
        dfb_rstd,
        mcast_ep,
        rstd_bytes,
        num_cores - 1,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = global_base},
        /*linked=*/true);
    bcast_sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_cores - 1);
    // The multicast excludes the root itself, which takes its copy locally.
    noc.async_write(
        dfb_rstd,
        remote_ep,
        rstd_bytes,
        {.offset_bytes = 0},
        {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = global_base});
    noc.async_write_barrier();
    bcast_sem.set(0);
    dfb_global.push_back(num_tile_rows);
    dfb_rstd.pop_front(num_tile_rows);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring reduce-scatter reduction. Port of reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp (no fused
// batching). Same ring state machine: 2-way adds in the middle steps, the 3-way add (input + interm + the other
// direction's partial) in the last step.
//
// Added hook (fuse_residual): in the last step the residual tile (cb_resid, read by the reader) is added on top of the
// 3-way sum with one dest-reuse add, so the writer emits partial_sum + residual: the residual add of the transformer
// layer, without another pass over the output.
//
// CT (A::): cb_input_id cb_interm_id cb_interm2_id cb_compute_output_id tile_granularity ring_size fuse_residual
//   cb_resid hw_startup traversals (batched: one ring traversal per batch, as the reader)
// RT: start_tiles_read start_tiles_to_read direction unit_start unit_end

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::rs {

template <typename A>
struct Reduction {
    uint32_t start_tiles_read, start_tiles_to_read, unit_start, unit_end;
    bool direction;

    void init() {
        start_tiles_read = rt<A>(0);
        start_tiles_to_read = rt<A>(1);
        direction = rt<A>(2);
        unit_start = rt<A>(3);
        unit_end = rt<A>(4);
    }

    void run() {
        constexpr uint32_t cb_input_id = A::cb_input_id;
        constexpr uint32_t cb_interm_id = A::cb_interm_id;
        constexpr uint32_t cb_interm2_id = A::cb_interm2_id;
        constexpr uint32_t cb_compute_output_id = A::cb_compute_output_id;
        constexpr uint32_t tile_granularity = A::tile_granularity;
        CircularBuffer cb_input(cb_input_id);
        CircularBuffer cb_interm(cb_interm_id);
        CircularBuffer cb_interm2(cb_interm2_id);
        CircularBuffer cb_compute_output(cb_compute_output_id);
        CircularBuffer cb_resid(A::cb_resid);

        if constexpr (A::hw_startup) {
            compute_kernel_hw_startup(cb_interm_id, cb_input_id, cb_compute_output_id);
        } else {
            reconfig_data_format(cb_interm_id, cb_input_id);
            pack_reconfig_data_format(cb_compute_output_id);
        }

        for (uint32_t t = 0; t < A::traversals; ++t) {
            // Units of this traversal: all of the worker's units, or (batched) its units of batch t.
            const uint32_t t_unit_start =
                A::traversals == 1 ? unit_start : (unit_start > t * A::slice_C ? unit_start : t * A::slice_C);
            const uint32_t t_unit_end =
                A::traversals == 1 ? unit_end : (unit_end < (t + 1) * A::slice_C ? unit_end : (t + 1) * A::slice_C);
            constexpr uint32_t ring_size_by_2 = A::ring_size / 2;
            uint32_t num_iters = ring_size_by_2 + 1;
            for (uint32_t i = 0; i < num_iters; ++i) {
                bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, reduce_output;
                if (i == 0) {
                    even_chunks = direction;
                    odd_chunks = !direction;
                    reduce_even_chunks = false;
                    reduce_odd_chunks = false;
                    reduce_output = false;
                } else if (i == ring_size_by_2) {
                    even_chunks = direction;
                    odd_chunks = !direction;
                    reduce_even_chunks = even_chunks;
                    reduce_odd_chunks = odd_chunks;
                    reduce_output = true;
                } else if (i == 1) {
                    even_chunks = true;
                    odd_chunks = true;
                    reduce_even_chunks = direction;
                    reduce_odd_chunks = !direction;
                    reduce_output = false;
                } else {
                    even_chunks = true;
                    odd_chunks = true;
                    reduce_even_chunks = even_chunks;
                    reduce_odd_chunks = odd_chunks;
                    reduce_output = false;
                }

                for (uint32_t u = t_unit_start; u < t_unit_end; ++u) {
                    uint32_t tiles_read = start_tiles_read;
                    uint32_t total_tiles_to_read = start_tiles_to_read;
                    while (tiles_read < total_tiles_to_read) {
                        const auto [is_even_chunk, tiles_to_read] =
                            reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);
                        if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                            tiles_read += tiles_to_read;
                        } else {
                            const bool reduce_interm =
                                (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);
                            if (reduce_interm) {
                                if (reduce_output) {
                                    cb_interm2.wait_front(tile_granularity);
                                }
                                cb_input.wait_front(tile_granularity);
                                cb_interm.wait_front(tile_granularity);

                                tile_regs_acquire();
                                if (reduce_output) {
                                    copy_init(cb_interm2_id);
                                    for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                        copy_tile(cb_interm2_id, tile_id, tile_id);
                                    }
                                    add_init(cb_interm_id, cb_input_id, true);  // DST = srcA + srcB + DST
                                } else {
                                    add_init(cb_interm_id, cb_input_id, false);  // DST = srcA + srcB
                                }
                                for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                    add_tiles(cb_interm_id, cb_input_id, tile_id, tile_id, tile_id);
                                }
                                if constexpr (A::fuse_residual) {
                                    if (reduce_output) {
                                        cb_resid.wait_front(tile_granularity);
                                        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(A::cb_resid);
                                        for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                                A::cb_resid, tile_id, tile_id);
                                        }
                                    }
                                }
                                tile_regs_commit();

                                if (reduce_output) {
                                    cb_interm2.pop_front(tile_granularity);
                                    if constexpr (A::fuse_residual) {
                                        cb_resid.pop_front(tile_granularity);
                                    }
                                }
                                cb_input.pop_front(tile_granularity);
                                cb_interm.pop_front(tile_granularity);

                                cb_compute_output.reserve_back(tile_granularity);
                                tile_regs_wait();
                                for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                                    pack_tile(tile_id, cb_compute_output_id, tile_id);
                                }
                                tile_regs_release();
                                cb_compute_output.push_back(tile_granularity);
                            }
                            tiles_read += tiles_to_read;
                        }
                    }
                }
            }
        }  // traversals
    }

    void teardown() {}
};

}  // namespace lego::rs

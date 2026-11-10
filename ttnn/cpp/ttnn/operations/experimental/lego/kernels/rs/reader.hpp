// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring reduce-scatter reader. Port of reduce_scatter_minimal_async/device/kernels/
// ring_reduce_scatter_minimal_async_reader.cpp, tiled staging (contiguous_interm = 0, which is what
// ttnn.reduce_scatter always selects) and no fused matmul receiver. Same ring state machine, chunk parity, semaphore
// waits and end-of-op semaphore reset.
//
// Added hook (fuse_residual): in the last ring step, next to each chunk's third term (read from the output tensor)
// the reader also reads the same output tiles of a residual tensor into cb_resid, for the reduction's residual add.
//
// CT (A::): my_chip_id ring_size cb_input_id cb_interm_id cb_interm2_id cb_reader_output_id tile_granularity
//   page_size input_batch_num_pages output_batch_num_pages input_channel_num_pages output_channel_num_pages
//   input_tensor_B input_tensor_Wt slice_C slice_Ht slice_Wt dim fuse_residual cb_resid traversals batch_wait
//   cta_input cta_interm cta_output cta_resid
// Common RT: the 8 ReduceScatterCommonArgs words (input intermediate output penult barrier sem0 sem1 ack), then
//   residual_addr when fuse_residual
// RT: direction chunks_per_sync start_tiles_read start_tiles_to_read start_pages_read_in_row start_row_offset
//   unit_start unit_end [batch_wait: batch_counters expected]
//
// Batched (traversals = input_tensor_B > 1, the original's fused-matmul mode): one ring traversal per batch, so the
// reduce-scatter of batch b overlaps the producer's batch b+1. With batch_wait the first step of a batch waits until
// L1 counter word 16 * b at batch_counters reaches `expected` (the producer's per-batch signal, in place of the
// original's ReduceScatterOpReceiver).

#pragma once

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/buffer_types.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::rs {

template <typename A>
struct Reader {
    using address_t = uint32_t;
    static constexpr uint32_t my_chip_id = A::my_chip_id;
    static constexpr uint32_t ring_size = A::ring_size;
    static constexpr uint32_t tile_granularity = A::tile_granularity;
    static constexpr uint32_t page_size = A::page_size;
    static constexpr uint32_t input_batch_num_pages = A::input_batch_num_pages;
    static constexpr uint32_t output_batch_num_pages = A::output_batch_num_pages;
    static constexpr uint32_t input_channel_num_pages = A::input_channel_num_pages;
    static constexpr uint32_t output_channel_num_pages = A::output_channel_num_pages;
    static constexpr uint32_t input_tensor_Wt = A::input_tensor_Wt;
    static constexpr uint32_t slice_C = A::slice_C;
    static constexpr uint32_t slice_Ht = A::slice_Ht;
    static constexpr uint32_t slice_Wt = A::slice_Wt;
    static constexpr uint32_t dim = A::dim;

    static FORCE_INLINE uint32_t slice_base_tile_id(uint32_t slice_idx) {
        if constexpr (dim == 3) {
            return slice_idx * slice_Wt;
        } else if constexpr (dim == 2) {
            return slice_idx * slice_Ht * slice_Wt;
        } else {
            return slice_idx * slice_C * slice_Ht * slice_Wt;
        }
    }

    // IntermSource</*Contiguous=*/false> of the original kernel.
    struct IntermSource {
        const uint32_t start_pages_read_in_row;
        const uint32_t start_row_offset;
        const uint32_t start_tiles_read;
        uint32_t interm_slice_base = 0;
        uint32_t output_batch_base = 0;
        uint32_t interm_tile_id_start = 0;
        uint32_t output_tile_id_start = 0;
        uint32_t pages_read_in_row = 0;
        uint32_t row_offset = 0;
        uint32_t output_tiles_read = 0;

        IntermSource(uint32_t pages_read_in_row_start, uint32_t row_offset_start, uint32_t tiles_read_start) :
            start_pages_read_in_row(pages_read_in_row_start),
            start_row_offset(row_offset_start),
            start_tiles_read(tiles_read_start) {}

        void begin_iteration(uint32_t b, uint32_t slice_idx) {
            interm_slice_base = b * input_batch_num_pages + slice_base_tile_id(slice_idx);
            output_batch_base = b * output_batch_num_pages;
        }
        void begin_channel(uint32_t c) {
            interm_tile_id_start = interm_slice_base + c * input_channel_num_pages;
            output_tile_id_start = output_batch_base + c * output_channel_num_pages;
            pages_read_in_row = start_pages_read_in_row;
            row_offset = start_row_offset;
            output_tiles_read = start_tiles_read;
        }
        void skip_chunk(uint32_t tiles) {
            for (uint32_t k = 0; k < tiles; ++k) {
                next_interm_tile_id();
                next_output_tile_id();
            }
        }
        template <typename IntermAcc, typename OutputAcc, typename ResidAcc>
        void read_chunk(
            Noc& noc,
            const IntermAcc& interm,
            const OutputAcc& output,
            const ResidAcc& resid,
            CircularBuffer& cb_interm,
            CircularBuffer& cb_interm2,
            CircularBuffer& cb_resid,
            uint32_t tiles_to_read,
            bool reduce_interm,
            bool reduce_output) {
            uint32_t interm_off = 0, interm2_off = 0;
            for (uint32_t j = 0; j < tiles_to_read; ++j) {
                const uint32_t interm_tile_id = next_interm_tile_id();
                const uint32_t output_tile_id = next_output_tile_id();
                if (!reduce_interm) {
                    continue;
                }
                noc.async_read(interm, cb_interm, page_size, {.page_id = interm_tile_id}, {.offset_bytes = interm_off});
                interm_off += page_size;
                if (reduce_output) {
                    noc.async_read(
                        output, cb_interm2, page_size, {.page_id = output_tile_id}, {.offset_bytes = interm2_off});
                    if constexpr (A::fuse_residual) {
                        noc.async_read(
                            resid, cb_resid, page_size, {.page_id = output_tile_id}, {.offset_bytes = interm2_off});
                    }
                    interm2_off += page_size;
                }
            }
        }

    private:
        uint32_t next_interm_tile_id() {
            uint32_t tile_id = interm_tile_id_start + row_offset + pages_read_in_row;
            ++pages_read_in_row;
            if (pages_read_in_row == slice_Wt) {
                row_offset += input_tensor_Wt;
                pages_read_in_row -= slice_Wt;
            }
            return tile_id;
        }
        uint32_t next_output_tile_id() { return output_tile_id_start + (output_tiles_read++); }
    };

    bool direction;
    size_t out_ready_sem, out2_ready_sem;
    uint32_t chunks_per_sync, start_tiles_to_read, start_pages_read_in_row, start_row_offset, unit_start, unit_end;
    int32_t start_tiles_read;
    address_t input_tensor_address, interm_tensor_address, output_tensor_address, resid_address;

    void init() {
        uint32_t arg_idx = 0;
        input_tensor_address = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::input);
        interm_tensor_address = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::intermediate);
        output_tensor_address = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::output);
        direction = rt<A>(arg_idx++);
        out_ready_sem = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::semaphore_0 + direction);
        out2_ready_sem = crt<A>(ttnn::ccl::ReduceScatterCommonArgs::semaphore_0 + !direction);
        chunks_per_sync = rt<A>(arg_idx++);
        start_tiles_read = rt<A>(arg_idx++);
        start_tiles_to_read = rt<A>(arg_idx++);
        start_pages_read_in_row = rt<A>(arg_idx++);
        start_row_offset = rt<A>(arg_idx++);
        unit_start = rt<A>(arg_idx++);
        unit_end = rt<A>(arg_idx++);
        resid_address = A::fuse_residual ? crt<A>(8) : 0;
        if constexpr (A::batch_wait) {
            batch_counters = rt<A>(arg_idx++);
            batch_expected = rt<A>(arg_idx++);
            // The producer's first batch signal comes long after kernel start.
            for (uint32_t b = 0; b < A::traversals; b++) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_counters + 16 * b)[0] = 0;
            }
        }
    }
    uint32_t batch_counters = 0, batch_expected = 0;
    void wait_batch(uint32_t b) {
        if constexpr (A::batch_wait) {
            LEGO_ZONE("rs_bwait");
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_counters + 16 * b), batch_expected);
        }
    }

    void run() {
        constexpr auto input_tensor_args = TensorAccessorArgs<A::cta_input>();
        auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
        constexpr auto interm_tensor_args = TensorAccessorArgs<A::cta_interm>();
        auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);
        constexpr auto output_tensor_args = TensorAccessorArgs<A::cta_output>();
        auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);
        constexpr auto resid_args = TensorAccessorArgs<A::cta_resid>();
        auto resid_accessor = TensorAccessor(resid_args, resid_address);

        IntermSource interm_source(start_pages_read_in_row, start_row_offset, start_tiles_read);

        Noc noc_obj;
        CircularBuffer cb_input(A::cb_input_id);
        CircularBuffer cb_interm(A::cb_interm_id);
        CircularBuffer cb_interm2(A::cb_interm2_id);
        CircularBuffer cb_reader_output(A::cb_reader_output_id);
        CircularBuffer cb_resid(A::cb_resid);

        uint32_t sem_target = 0;
        uint32_t sem2_target = 0;

        for (uint32_t t = 0; t < A::traversals; ++t) {
            // Units of this traversal: all of the worker's units, or (batched) its units of batch t.
            const uint32_t t_unit_start =
                A::traversals == 1 ? unit_start : (unit_start > t * A::slice_C ? unit_start : t * A::slice_C);
            const uint32_t t_unit_end =
                A::traversals == 1 ? unit_end : (unit_end < (t + 1) * A::slice_C ? unit_end : (t + 1) * A::slice_C);
            constexpr uint32_t ring_size_by_2 = ring_size / 2;
            int slice_idx = my_chip_id + ring_size_by_2;  // start with the slice of the device half-way across the ring
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
                if (slice_idx < 0) {
                    slice_idx += ring_size;
                } else if (slice_idx >= (int)ring_size) {
                    slice_idx = (uint32_t)slice_idx - ring_size;
                }
                const uint32_t slice_base = slice_base_tile_id(slice_idx);

                uint32_t input_tile_id_start = 0;
                uint32_t input_pages_read_in_row = start_pages_read_in_row;
                uint32_t input_row_offset = start_row_offset;
                auto get_next_input_tile_id = [&]() -> uint32_t {
                    uint32_t tile_id = input_tile_id_start + input_row_offset + input_pages_read_in_row;
                    ++input_pages_read_in_row;
                    if (input_pages_read_in_row == slice_Wt) {
                        input_row_offset += input_tensor_Wt;
                        input_pages_read_in_row -= slice_Wt;
                    }
                    return tile_id;
                };

                uint32_t chunk_count = 0;
                for (uint32_t u = t_unit_start; u < t_unit_end; ++u) {
                    const uint32_t b = u / slice_C;
                    const uint32_t c = u % slice_C;
                    if (i == 0) {
                        wait_batch(b);
                    }
                    input_tile_id_start = slice_base + b * input_batch_num_pages + c * input_channel_num_pages;
                    input_pages_read_in_row = start_pages_read_in_row;
                    input_row_offset = start_row_offset;
                    interm_source.begin_iteration(b, slice_idx);
                    interm_source.begin_channel(c);
                    uint32_t tiles_read = start_tiles_read;
                    uint32_t total_tiles_to_read = start_tiles_to_read;

                    while (tiles_read < total_tiles_to_read) {
                        const auto [is_even_chunk, tiles_to_read] =
                            reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);

                        if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                            tiles_read += tiles_to_read;
                            for (uint32_t k = 0; k < tiles_to_read; ++k) {
                                get_next_input_tile_id();
                            }
                            interm_source.skip_chunk(tiles_to_read);
                        } else {
                            const bool reduce_interm =
                                (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);
                            CircularBuffer& cb_in = reduce_interm ? cb_input : cb_reader_output;

                            if (reduce_interm) {
                                if (chunk_count == 0) {
                                    noc_semaphore_wait_min(
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                                    ++sem_target;
                                    if (reduce_output) {
                                        noc_semaphore_wait_min(
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out2_ready_sem),
                                            sem2_target + 1);
                                        ++sem2_target;
                                    }
                                }
                                chunk_count = (chunk_count == chunks_per_sync - 1) ? 0 : (chunk_count + 1);
                            }

                            cb_in.reserve_back(tile_granularity);
                            uint32_t l1_write_offset = 0;
                            if (reduce_interm) {
                                cb_interm.reserve_back(tile_granularity);
                                if (reduce_output) {
                                    cb_interm2.reserve_back(tile_granularity);
                                    if constexpr (A::fuse_residual) {
                                        cb_resid.reserve_back(tile_granularity);
                                    }
                                }
                            }
                            for (uint32_t j = 0; j < tiles_to_read; ++j) {
                                auto input_tile_id = get_next_input_tile_id();
                                noc_obj.async_read(
                                    input_tensor_accessor,
                                    cb_in,
                                    page_size,
                                    {.page_id = input_tile_id},
                                    {.offset_bytes = l1_write_offset});
                                l1_write_offset += page_size;
                            }
                            interm_source.read_chunk(
                                noc_obj,
                                interm_tensor_accessor,
                                output_tensor_accessor,
                                resid_accessor,
                                cb_interm,
                                cb_interm2,
                                cb_resid,
                                tiles_to_read,
                                reduce_interm,
                                reduce_output);
                            tiles_read += tiles_to_read;
                            noc_obj.async_read_barrier();
                            cb_in.push_back(tile_granularity);
                            if (reduce_interm) {
                                cb_interm.push_back(tile_granularity);
                                if (reduce_output) {
                                    cb_interm2.push_back(tile_granularity);
                                    if constexpr (A::fuse_residual) {
                                        cb_resid.push_back(tile_granularity);
                                    }
                                }
                            }
                        }
                    }
                }
                slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
            }
        }  // traversals
    }

    void teardown() {
        // Reset the out_ready semaphores once, after the whole ring traversal
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out2_ready_sem), 0);
    }
};

}  // namespace lego::rs

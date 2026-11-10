// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring all-gather reader. Port of all_gather_async/device/kernels/minimal_default_reader.cpp (interleaved input and
// output, no fused-op signaller): stream the local slice to the writer, then for each incoming slice wait on the
// out_ready semaphore per chunks_per_sync packet group and, for the slices this worker forwards, read them back
// from the output tensor for the writer. Same split forwarding on even rings.
//
// CT (A::): ring_size my_chip_id cb_output_id num_tiles_to_write_per_packet page_size
//   num_targets_forward_direction num_targets_backward_direction topology gather_dim input_batch_head_count
//   input_tensor_Wt input_tensor_C output_tensor_Wt output_tensor_C reverse cta_input cta_output
// Added hook (signal_slices), the role of the original's fused-op signaller: after this worker has received its part
// of a slice, increment the L1 counter word (16 * slice position) at slice_counters on each target core (e.g. the
// injectors of a matmul that consumes the gathered tensor slice by slice).
//
// With row_group_tiles > 0 the signal is per row group instead: target i owns slice tiles
// [i * row_group_tiles, (i + 1) * row_group_tiles) (e.g. one matmul injector's rows), and it is signalled as soon as
// this worker's tiles in that range have landed (checked after every out_ready wait), not at the end of the slice.
//
// Common RT: the 6 AllGatherCommonArgs words (input output barrier sem0 sem1 input_Ht)
// RT: direction global_worker_id global_worker_count chunks_per_sync_or_zero
//   [signal_slices: slice_counters num_targets x0 y0 x1 y1 ...]

#pragma once

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/kernels/all_gather_worker_split.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::ag {

template <typename A>
struct Reader {
    using address_t = uint32_t;
    using Topology = ttnn::ccl::Topology;
    static constexpr uint32_t ring_size = A::ring_size;
    static constexpr uint32_t my_chip_id = A::my_chip_id;
    static constexpr uint32_t num_tiles_to_write_per_packet = A::num_tiles_to_write_per_packet;
    static constexpr uint32_t page_size = A::page_size;
    static constexpr uint32_t num_targets_forward_direction = A::num_targets_forward_direction;
    static constexpr uint32_t num_targets_backward_direction = A::num_targets_backward_direction;
    static constexpr Topology topology = static_cast<Topology>(A::topology);
    static constexpr uint32_t gather_dim = A::gather_dim;
    static constexpr uint32_t input_batch_head_count = A::input_batch_head_count;
    static constexpr uint32_t input_tensor_Wt = A::input_tensor_Wt;
    static constexpr uint32_t input_tensor_C = A::input_tensor_C;
    static constexpr uint32_t output_tensor_Wt = A::output_tensor_Wt;
    static constexpr uint32_t output_tensor_C = A::output_tensor_C;
    static constexpr bool reverse = A::reverse;

    address_t input_tensor_address, output_tensor_address;
    uint32_t input_tensor_Ht, output_tensor_Ht;
    bool direction;
    size_t out_ready_sem;
    AllGatherWorkerSplit split;
    uint32_t sem_target = 0;
    uint32_t sig_arg = 0;

    // Row-group signalling state for the slice being received: this worker's tiles [p0, p1) of it, and the first group
    // not yet signalled.
    uint32_t p0 = 0, p1 = 0, next_group = 0, cur_slice = 0;

    // Row group t has signal_copies targets (e.g. the matching injectors of several matmuls that split one by N).
    void signal_target(uint32_t t, uint32_t slice) {
        const uint32_t counters = rt<A>(sig_arg);
        for (uint32_t k = 0; k < A::signal_copies; k++) {
            const uint32_t i = t * A::signal_copies + k;
            noc_semaphore_inc(
                get_noc_addr(rt<A>(sig_arg + 2 + 2 * i), rt<A>(sig_arg + 3 + 2 * i), counters + 16 * slice), 1);
        }
    }
    void begin_portion(uint32_t slice, uint32_t start, uint32_t end) {
        if constexpr (A::signal_slices && A::row_group_tiles > 0) {
            cur_slice = slice;
            p0 = start;
            p1 = end;
            next_group = start / A::row_group_tiles;
        }
    }
    // Tiles [p0, landed) of the current portion have landed: signal every group they complete.
    void progress(uint32_t landed) {
        if constexpr (A::signal_slices && A::row_group_tiles > 0) {
            if (landed > p1) {
                landed = p1;
            }
            const uint32_t n = rt<A>(sig_arg + 1);
            while (next_group < n) {
                const uint32_t g_end = (next_group + 1) * A::row_group_tiles;
                if (landed < g_end && landed < p1) {
                    break;
                }
                if (next_group * A::row_group_tiles >= p1) {
                    break;
                }
                signal_target(next_group, cur_slice);
                next_group++;
            }
        }
    }

    void signal_slice(uint32_t slice) {
        if constexpr (A::signal_slices && A::row_group_tiles > 0) {
            progress(p1);
        } else if constexpr (A::signal_slices) {
            noc_async_read_barrier();
            const uint32_t counters = rt<A>(sig_arg);
            const uint32_t n = rt<A>(sig_arg + 1);
            for (uint32_t t = 0; t < n; t++) {
                noc_semaphore_inc(
                    get_noc_addr(rt<A>(sig_arg + 2 + 2 * t), rt<A>(sig_arg + 3 + 2 * t), counters + 16 * slice), 1);
            }
        }
    }

    void init() {
        uint32_t arg_idx = 0;
        input_tensor_address = crt<A>(ttnn::ccl::AllGatherCommonArgs::input);
        output_tensor_address = crt<A>(ttnn::ccl::AllGatherCommonArgs::output);
        input_tensor_Ht = crt<A>(ttnn::ccl::AllGatherCommonArgs::input_Ht);
        output_tensor_Ht = gather_dim == 2 ? input_tensor_Ht * ring_size : input_tensor_Ht;
        direction = rt<A>(arg_idx++);
        out_ready_sem = crt<A>(ttnn::ccl::AllGatherCommonArgs::semaphore_0 + direction);
        const auto global_worker_id = rt<A>(arg_idx++);
        const auto global_worker_count = rt<A>(arg_idx++);
        const auto chunks_per_sync_or_zero = rt<A>(arg_idx++);
        split = all_gather_worker_split(
            global_worker_id,
            global_worker_count,
            input_tensor_Ht,
            input_tensor_Wt,
            output_tensor_Wt,
            num_tiles_to_write_per_packet,
            chunks_per_sync_or_zero);
        sig_arg = arg_idx;
    }

    void run() {
        const uint32_t input_tile_id_start = split.input_tile_id_start;
        const uint32_t input_tile_id_end = split.input_tile_id_end;
        const uint32_t start_pages_read_in_row = split.start_pages_read_in_row;
        const uint32_t start_row_offset = split.start_row_offset;
        const uint32_t chunks_per_sync = split.chunks_per_sync;

        constexpr auto input_tensor_args = TensorAccessorArgs<A::cta_input>();
        const auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address);
        constexpr auto output_tensor_args = TensorAccessorArgs<A::cta_output>();
        const auto output_tensor_addrgen = TensorAccessor(output_tensor_args, output_tensor_address);

        Noc noc_obj;
        CircularBuffer cb_output(A::cb_output_id);

        uint32_t tiles_read = input_tile_id_start;
        uint32_t tiles_to_read = input_tile_id_end;
        uint32_t output_tile_id_start = 0;
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                cb_output.reserve_back(num_tiles_to_write_per_packet);
                size_t l1_write_addr = cb_output.get_write_ptr();
                for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                    uint32_t tile_id = output_tile_id_start + tiles_read;
                    uint64_t noc_read_addr = input_tensor_addrgen.get_noc_addr(tile_id);
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                    l1_write_addr += page_size;
                    tiles_read++;
                }
                noc_obj.async_read_barrier();
                cb_output.push_back(num_tiles_to_write_per_packet);
            }
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;
            output_tile_id_start += input_tensor_Wt * input_tensor_Ht;
        }

        uint32_t slices_received = 0;
        uint32_t slices_expected = 0;
        uint32_t writes_expected = 0;
        if constexpr (topology == Topology::Linear) {
            if (direction == 1) {
                slices_expected = num_targets_forward_direction;
                writes_expected = num_targets_backward_direction ? num_targets_forward_direction : 0;
            } else {
                slices_expected = num_targets_backward_direction;
                writes_expected = num_targets_forward_direction ? num_targets_backward_direction : 0;
            }
        } else if constexpr (topology == Topology::Ring) {
            if (direction == 1) {
                slices_expected = num_targets_backward_direction;
                writes_expected = num_targets_backward_direction - 1;
            } else {
                slices_expected = num_targets_forward_direction;
                writes_expected = num_targets_forward_direction - 1;
            }
        }
        bool split_forwarding_enabled = false;
        if constexpr (topology == Topology::Ring) {
            if (ring_size % 2 == 0 && ring_size > 2 && input_tile_id_end - input_tile_id_start >= 2) {
                split_forwarding_enabled = true;
                if (direction == 1) {
                    slices_expected++;
                    writes_expected++;
                }
            }
        }

        uint32_t chunk_count = 0;
        uint32_t slices_forwarded = 0;
        while (slices_received < slices_expected) {
            bool is_last_slice = (slices_received == slices_expected - 1);
            bool is_split_slice = split_forwarding_enabled && is_last_slice;
            int sender_chip_id;
            uint32_t actual_sender_chip_id;
            if (direction == 1) {
                sender_chip_id = my_chip_id + slices_received + 1;
                actual_sender_chip_id =
                    (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
            } else {
                sender_chip_id = my_chip_id - (slices_received + 1);
                actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
            }
            if constexpr (reverse) {
                actual_sender_chip_id = (ring_size - 1) - actual_sender_chip_id;
            }
            bool should_forward = false;
            if constexpr (topology == Topology::Linear) {
                should_forward = (writes_expected > 0);
            } else if constexpr (topology == Topology::Ring) {
                should_forward = ((slices_received + 1) < (writes_expected + 1));
            }
            bool is_last_forwarded_slice = should_forward && (slices_forwarded == writes_expected - 1);
            bool is_split_forwarded_slice = split_forwarding_enabled && is_last_forwarded_slice;

            if (should_forward) {
                uint32_t out_tile_id_start = 0;
                uint32_t slice_Wt = input_tensor_Wt;
                uint32_t stride_Wt = output_tensor_Wt;
                uint32_t total_tiles = input_tile_id_end - input_tile_id_start;
                uint32_t first_half_tiles = is_split_forwarded_slice ? total_tiles / 2 : 0;
                if constexpr (gather_dim == 3) {
                    out_tile_id_start = actual_sender_chip_id * input_tensor_Wt;
                } else if constexpr (gather_dim == 2) {
                    out_tile_id_start = actual_sender_chip_id * input_tensor_Ht * input_tensor_Wt;
                } else if constexpr (gather_dim == 1) {
                    out_tile_id_start = actual_sender_chip_id * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
                } else {
                    out_tile_id_start =
                        actual_sender_chip_id * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
                }
                uint32_t num_channels_processed_in_current_batch = 0;
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                    chunk_count = 0;
                    uint32_t cb_row_offset = start_row_offset;
                    uint32_t cb_pages_read_in_row = start_pages_read_in_row;
                    if (is_split_forwarded_slice && direction == 1) {
                        uint32_t tiles_to_skip = first_half_tiles;
                        while (tiles_to_skip > 0) {
                            if (tiles_to_skip < slice_Wt - cb_pages_read_in_row) {
                                cb_pages_read_in_row += tiles_to_skip;
                                tiles_to_skip = 0;
                            } else {
                                tiles_to_skip -= (slice_Wt - cb_pages_read_in_row);
                                cb_row_offset += stride_Wt;
                                cb_pages_read_in_row = 0;
                            }
                        }
                    }
                    uint32_t cb_tile_start = is_split_forwarded_slice && direction == 1
                                                 ? input_tile_id_start + first_half_tiles
                                                 : input_tile_id_start;
                    uint32_t cb_tile_end = is_split_forwarded_slice && direction == 0
                                               ? input_tile_id_start + first_half_tiles
                                               : input_tile_id_end;
                    uint32_t cb_tiles_pushed = cb_tile_start;
                    uint32_t sem_iter_pos = input_tile_id_start;
                    begin_portion(actual_sender_chip_id, input_tile_id_start, input_tile_id_end);
                    uint32_t waits = 0;
                    while (sem_iter_pos < input_tile_id_end) {
                        if (chunk_count % chunks_per_sync == 0) {
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                            sem_target++;
                            waits++;
                            progress(input_tile_id_start + waits * chunks_per_sync * num_tiles_to_write_per_packet);
                        }
                        chunk_count++;
                        uint32_t tiles_remaining = input_tile_id_end - sem_iter_pos;
                        sem_iter_pos += std::min(tiles_remaining, num_tiles_to_write_per_packet);
                        while (cb_tiles_pushed < cb_tile_end) {
                            uint32_t cb_tiles_remaining = cb_tile_end - cb_tiles_pushed;
                            uint32_t num_tiles_to_read = std::min(cb_tiles_remaining, num_tiles_to_write_per_packet);
                            if (cb_tiles_pushed + num_tiles_to_read <= sem_iter_pos) {
                                cb_output.reserve_back(num_tiles_to_write_per_packet);
                                size_t l1_write_addr = cb_output.get_write_ptr();
                                for (uint32_t j = 0; j < num_tiles_to_read; ++j) {
                                    uint32_t tile_id = out_tile_id_start + cb_row_offset + cb_pages_read_in_row;
                                    uint64_t noc_read_addr = output_tensor_addrgen.get_noc_addr(tile_id);
                                    noc_async_read(noc_read_addr, l1_write_addr, page_size);
                                    l1_write_addr += page_size;
                                    cb_pages_read_in_row++;
                                    if (cb_pages_read_in_row >= slice_Wt) {
                                        cb_row_offset += stride_Wt;
                                        cb_pages_read_in_row = 0;
                                    }
                                }
                                noc_obj.async_read_barrier();
                                cb_output.push_back(num_tiles_to_write_per_packet);
                                cb_tiles_pushed += num_tiles_to_read;
                            } else {
                                break;
                            }
                        }
                    }
                    num_channels_processed_in_current_batch++;
                    if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
                        out_tile_id_start +=
                            output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
                    } else {
                        out_tile_id_start += output_tensor_Wt * output_tensor_Ht;
                    }
                    if (num_channels_processed_in_current_batch == input_tensor_C) {
                        num_channels_processed_in_current_batch = 0;
                    }
                }
                slices_forwarded++;
            } else {
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                    chunk_count = 0;
                    tiles_read = input_tile_id_start;
                    tiles_to_read = input_tile_id_end;
                    if (is_split_slice) {
                        uint32_t total_tiles = input_tile_id_end - input_tile_id_start;
                        uint32_t first_half_tiles = total_tiles / 2;
                        if (direction == 0) {
                            tiles_to_read = input_tile_id_start + first_half_tiles;
                        } else {
                            tiles_read = input_tile_id_start + first_half_tiles;
                        }
                    }
                    begin_portion(actual_sender_chip_id, tiles_read, tiles_to_read);
                    const uint32_t portion_start = tiles_read;
                    uint32_t waits = 0;
                    while (tiles_read < tiles_to_read) {
                        if (chunk_count % chunks_per_sync == 0) {
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                            sem_target++;
                            waits++;
                            progress(portion_start + waits * chunks_per_sync * num_tiles_to_write_per_packet);
                        }
                        chunk_count++;
                        uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                        uint32_t num_tiles_to_read = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                        tiles_read += num_tiles_to_read;
                    }
                }
            }
            slices_received++;
            signal_slice(actual_sender_chip_id);
        }
        // Return the credits this reader consumed (the semaphore is global and persists across calls).
        noc_semaphore_inc(get_noc_addr(out_ready_sem), uint32_t{0} - sem_target);
        noc_obj.async_write_barrier();
        noc_obj.async_atomic_barrier();
    }

    void teardown() {}
};

}  // namespace lego::ag

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/paged_kv_utils.hpp"
#include "ring_attention_all_gather_metadata.hpp"
#include "ring_attention_rank_mapping.hpp"
#include "ring_attention_prefetch_utils.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t my_transport_rank = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);  // 2
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(6));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(7);  // 2
constexpr uint32_t num_inputs = get_compile_time_arg_val(8);
constexpr bool direction = get_compile_time_arg_val(9);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(10);
constexpr bool has_metadata = get_compile_time_arg_val(11);
constexpr uint32_t cb_meta_id = get_compile_time_arg_val(12);
constexpr uint32_t num_links = get_compile_time_arg_val(13);
// Host-derived even-ring split-forwarding gate: the parent fused op owns this protocol decision and
// passes the same flag to both all-gather directions and its own receiver, so producer and consumer
// cannot disagree. Standalone (non-fused) callers get the legacy even-ring topology gate from the host.
constexpr bool split_forwarding_enabled = get_compile_time_arg_val(14);
constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(15);
constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(16));
constexpr uint32_t mesh_rows = get_compile_time_arg_val(17);
constexpr uint32_t mesh_cols = get_compile_time_arg_val(18);
constexpr bool has_page_bundles = get_compile_time_arg_val(19);
constexpr uint32_t cb_page_bundle_id = get_compile_time_arg_val(20);

// Prefetch: batch multiple packets of DRAM reads before a single barrier.
// This keeps more reads in flight across interleaved DRAM banks, hiding latency.
// CB depth must be >= 2 * PREFETCH_PACKETS * packet_size_in_pages (see program_factory cb_num_pages).
constexpr uint32_t PREFETCH_PACKETS = 4;

void kernel_main() {
    constexpr uint32_t page_size_base_idx = ttnn::ring_attention_all_gather::kReaderFixedCompileTimeArgCount;
    constexpr auto inputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<
        num_inputs,
        std::get<num_inputs - 1>(inputs_args).next_compile_time_args_offset()>();
    constexpr uint32_t kMetaArgsOffset = has_metadata
                                             ? std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset()
                                             : (page_size_base_idx + num_inputs);
    constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
    constexpr uint32_t kKvMetaArgsOffset = has_metadata ? meta_args.next_compile_time_args_offset() : kMetaArgsOffset;
    constexpr auto kv_meta_args = TensorAccessorArgs<kKvMetaArgsOffset>();
    constexpr uint32_t kPostMetaArgsOffset =
        has_metadata ? kv_meta_args.next_compile_time_args_offset()
                     : std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset();
    constexpr uint32_t kPageBundleArgsOffset =
        has_page_bundles ? kPostMetaArgsOffset : std::get<0>(inputs_args).next_compile_time_args_offset();
    constexpr auto page_bundle_args = TensorAccessorArgs<kPageBundleArgsOffset>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_tensor_Wt;
    std::array<uint32_t, num_inputs> input_tensor_Ht;
    std::array<uint32_t, num_inputs> output_tensor_Wt;
    std::array<uint32_t, num_inputs> output_tensor_Ht;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_id_start;
    std::array<uint32_t, num_inputs> input_tile_id_end;
    std::array<uint32_t, num_inputs> input_valid_pages;
    std::array<uint32_t, num_inputs> worker_link;
    // Phase-1 input page base: nonzero only for single-slot gather (skip to the sliced input slot).
    // The slice is always emitted into output slot 0, whatever the output batch size.
    std::array<uint32_t, num_inputs> input_batch_base;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        (void)get_arg_val<uint32_t>(arg_idx++);  // structural tile_id_start placeholder
        (void)get_arg_val<uint32_t>(arg_idx++);  // structural tile_id_end placeholder
        input_batch_base[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        // valid_pages_per_batch_head (slot 8): clamp the gather to the logical_n-valid slab prefix so
        // only kv_actual-sized data moves. Uniform across cores/devices, so producer/consumer page
        // counts and the ring slice protocol stay matched. Default (full input) leaves it unchanged.
        const uint32_t valid_pages = get_arg_val<uint32_t>(arg_idx++);
        worker_link[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_valid_pages[input_idx] = valid_pages;
        const auto link_page_range =
            ring_attention_all_gather::compute_link_page_range(valid_pages, num_links, worker_link[input_idx]);
        input_tile_id_start[input_idx] = link_page_range.start;
        input_tile_id_end[input_idx] = link_page_range.end;
    }

    auto inputs_tuple = make_tensor_accessor_tuple(inputs_args, arg_idx);
    arg_idx += num_inputs;
    auto input_tensor_addrgens = make_abstract_tensor_accessor_wrappers(inputs_tuple);
    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_tensor_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);

    if constexpr (has_metadata) {
        const uint32_t slot_id_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_actual_isl_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t chunk_local_tiles = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_cache_num_layers = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_cache_layer_idx = get_arg_val<uint32_t>(arg_idx++);
        Noc meta_noc;
        // Use the data CB as temporary metadata scratch. It is empty at this point and avoids
        // a separate tiny-CB read race on the all-gather worker cores.
        CircularBuffer cb_meta(cb_output_id);
        const uint32_t slot_id =
            trace_metadata::read_metadata_scalar_u32(meta_noc, meta_args, slot_id_addr, cb_meta.get_write_ptr());
        const uint32_t cache_batch_idx = slot_id * kv_cache_num_layers + kv_cache_layer_idx;
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            input_batch_base[input_idx] = cache_batch_idx * input_batch_head_count[input_idx] *
                                          input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
        }
        const uint32_t kv_actual = trace_metadata::read_metadata_scalar_u32(
            meta_noc, kv_meta_args, kv_actual_isl_addr, cb_meta.get_write_ptr());
        const uint32_t gather_valid_Ht =
            ring_attention_all_gather::compute_gather_valid_Ht(kv_actual, chunk_local_tiles, ring_size);
        ring_attention_all_gather::update_link_page_ranges_for_gather_extent(
            gather_valid_Ht,
            num_links,
            input_tensor_Wt,
            input_valid_pages,
            worker_link,
            input_tile_id_start,
            input_tile_id_end);
    }

    uint32_t page_bundle_indices_addr = 0;
    uint32_t page_bundle_num_layers = 1;
    uint32_t page_bundle_layer_idx = 0;
    uint32_t page_bundle_size_tiles = 1;
    if constexpr (has_page_bundles) {
        page_bundle_indices_addr = get_arg_val<uint32_t>(arg_idx++);
        page_bundle_num_layers = get_arg_val<uint32_t>(arg_idx++);
        page_bundle_layer_idx = get_arg_val<uint32_t>(arg_idx++);
        page_bundle_size_tiles = get_arg_val<uint32_t>(arg_idx++);
    }

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);
    const auto page_bundle_reader = TensorAccessor(page_bundle_args, page_bundle_indices_addr);
    uint32_t page_bundle_scratch = 0;
    if constexpr (has_page_bundles) {
        CircularBuffer cb_page_bundle(cb_page_bundle_id);
        page_bundle_scratch = cb_page_bundle.get_write_ptr();
        noc_obj.async_read(
            page_bundle_reader,
            CoreLocalMem<uint16_t>(page_bundle_scratch),
            (input_tensor_Ht[0] / page_bundle_size_tiles) * sizeof(uint16_t),
            {.page_id = 0},
            {});
        noc_obj.async_read_barrier();
        invalidate_l1_cache();
    }

    // Push out our local slice
    // For a single-slot gather this starts at the sliced batch slot; otherwise 0 (full batch).
    uint32_t output_tile_id_start = 0;
    // Read local slice to our buffers, before sending them over
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        output_tile_id_start = input_batch_base[input_idx];
        uint32_t tiles_read = input_tile_id_start[input_idx];
        uint32_t tiles_to_read = input_tile_id_end[input_idx];
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
            uint32_t page_bundle_col = 0;
            PagedKVBundleCursor bundle_cursor;
            uint32_t bundle_stride = 0;
            uint32_t head_offset = 0;
            uint32_t physical_bundle_offset = 0;
            if constexpr (has_page_bundles) {
                page_bundle_col = tiles_read % input_tensor_Wt[input_idx];
                const uint32_t first_logical_row = tiles_read / input_tensor_Wt[input_idx];
                bundle_cursor.reset(page_bundle_scratch, first_logical_row, page_bundle_size_tiles);
                bundle_stride = page_bundle_num_layers * input_batch_head_count[input_idx] * page_bundle_size_tiles *
                                input_tensor_Wt[input_idx];
                head_offset = (page_bundle_layer_idx * input_batch_head_count[input_idx] + bh_idx) *
                              page_bundle_size_tiles * input_tensor_Wt[input_idx];
                physical_bundle_offset = bundle_cursor.physical_bundle() * bundle_stride;
            }
            prefetch_batch_read_tiles<
                input_tensor_page_size,
                packet_size_in_pages,
                PREFETCH_PACKETS,
                contig_pages_advanced>(
                noc_obj,
                cb_output,
                tiles_read,
                tiles_to_read,
                cb_fifo_limit,
                cb_fifo_size,
                input_tensor_addrgens[input_idx],
                [&](uint32_t tr) {
                    if constexpr (!has_page_bundles) {
                        return output_tile_id_start + tr;
                    } else {
                        [[maybe_unused]] const uint32_t page_id =
                            physical_bundle_offset + head_offset +
                            bundle_cursor.row_in_bundle * input_tensor_Wt[input_idx] + page_bundle_col;
                        page_bundle_col += contig_pages_advanced;
                        while (page_bundle_col >= input_tensor_Wt[input_idx]) {
                            page_bundle_col -= input_tensor_Wt[input_idx];
                            if (bundle_cursor.advance_row()) {
                                // Avoid loading one entry beyond the table after the final transfer.
                                if (tr + contig_pages_advanced < tiles_to_read) {
                                    physical_bundle_offset = bundle_cursor.physical_bundle() * bundle_stride;
                                }
                            }
                        }
                        return page_id;
                    }
                });
            tiles_read = input_tile_id_start[input_idx];
            tiles_to_read = input_tile_id_end[input_idx];
            output_tile_id_start += input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
        }
        output_tile_id_start = 0;
    }

    uint32_t slices_received = 0;
    uint32_t slices_expected = 0;
    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1) {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_backward_direction ? num_targets_forward_direction : 0;
        } else {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_forward_direction ? num_targets_backward_direction : 0;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_backward_direction - 1;
        } else {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    // Mirror the writer's split-forwarding (see ring_attention_all_gather_writer.cpp): on an even ring the diametric
    // slice is relayed half per direction. The gate is a host-derived compile-time flag (see top of file).
    if (split_forwarding_enabled && direction == 1) {
        slices_expected++;
        writes_expected++;
    }

    while (slices_received < slices_expected) {
        // Do i expect more from the backward direction?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right, (keep in mind this differs
        // for odd/even chips)
        // Do i expect more from the forward direction?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)

        // Device 2.0: legacy primitive retained, out_ready_sem is the address of a GlobalSemaphore
        // Semaphore<> binds to per-program ids via get_semaphore<>(id), so it cannot wrap a
        // GlobalSemaphore.
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), slices_received + 1);
        // Got it
        slices_received++;

        int sender_transport_rank_signed;
        uint32_t sender_transport_rank;
        if constexpr (direction == 1) {
            sender_transport_rank_signed = my_transport_rank + slices_received;
            sender_transport_rank = (sender_transport_rank_signed >= (int)ring_size)
                                        ? sender_transport_rank_signed - ring_size
                                        : sender_transport_rank_signed;
        } else {
            sender_transport_rank_signed = my_transport_rank - slices_received;
            sender_transport_rank = (sender_transport_rank_signed < 0) ? ring_size + sender_transport_rank_signed
                                                                       : sender_transport_rank_signed;
        }
        const uint32_t sender_tensor_rank =
            ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                sender_transport_rank, mesh_rows, mesh_cols, snake_orientation);

        if constexpr (fuse_op) {
            // Signal matmul to go
            op_signaler.synchronize_workers_and_signal_op(sender_transport_rank);
        }
        // Direction == backward: Should I forward what I got from the left to my right?
        // In the linear case, if I have any targets to my right, always forward
        // In the ring case, if I have received on the left less than my targets on the right, forward
        // Direction == forward: Should I forward what I got from the right to my left?
        // In the linear case, if I have any targets to my left, always forward
        // In the ring case, if I have received on the right less than my targets on the left, forward
        if ((topology == Topology::Linear && writes_expected > 0) ||
            (topology == Topology::Ring && (slices_received < (writes_expected + 1)))) {
            // The last slice we relay is the diametric shard of our downstream neighbor
            const bool is_split_forwarded_slice = split_forwarding_enabled && (slices_received == writes_expected);
            for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
                uint32_t slice_Wt = input_tensor_Wt[input_idx];
                uint32_t stride_Wt = output_tensor_Wt[input_idx];

                // Packet-aligned midpoint of this input's per-batch-head page range (matches the writer)
                const uint32_t total_pages = input_tile_id_end[input_idx] - input_tile_id_start[input_idx];
                const uint32_t num_packets = (total_pages + packet_size_in_pages - 1) / packet_size_in_pages;
                const uint32_t first_half_pages = (num_packets / 2) * packet_size_in_pages;
                const bool split_this_input = is_split_forwarded_slice;
                uint32_t relay_start = input_tile_id_start[input_idx];
                uint32_t relay_end = input_tile_id_end[input_idx];
                if (split_this_input) {
                    if (direction == 0) {
                        relay_end = input_tile_id_start[input_idx] + first_half_pages;
                    } else {
                        relay_start = input_tile_id_start[input_idx] + first_half_pages;
                    }
                }

                uint32_t tiles_read = relay_start;
                uint32_t tiles_to_read = relay_end;
                uint32_t output_tile_id_start = 0;
                uint32_t pages_read_in_row = relay_start % slice_Wt;
                uint32_t row_offset = (relay_start / slice_Wt) * stride_Wt;
                if (gather_dim == 3) {
                    output_tile_id_start = sender_tensor_rank * input_tensor_Wt[input_idx];
                } else {
                    output_tile_id_start = sender_tensor_rank * input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                    prefetch_batch_read_tiles<
                        input_tensor_page_size,
                        packet_size_in_pages,
                        PREFETCH_PACKETS,
                        contig_pages_advanced>(
                        noc_obj,
                        cb_output,
                        tiles_read,
                        tiles_to_read,
                        cb_fifo_limit,
                        cb_fifo_size,
                        output_tensor_addrgens[input_idx],
                        [&](uint32_t /* tiles_read */) {
                            const uint32_t pid = output_tile_id_start + row_offset + pages_read_in_row;
                            pages_read_in_row++;
                            if (pages_read_in_row >= slice_Wt) {
                                row_offset += stride_Wt;
                                pages_read_in_row = 0;
                            }
                            return pid;
                        });
                    pages_read_in_row = relay_start % slice_Wt;
                    row_offset = (relay_start / slice_Wt) * stride_Wt;
                    tiles_read = relay_start;
                    tiles_to_read = relay_end;
                    output_tile_id_start += output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
                }
            }
        }
    }

    // Flush non-posted atomics from op_signaler before kernel exit (mirrors the writer's barrier).
    if constexpr (fuse_op) {
        noc_obj.async_atomic_barrier();
    }
    // Device 2.0 migration: legacy primitive retained, out_ready_sem is a GlobalSemaphore address.
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}

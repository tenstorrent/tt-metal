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
#include "ring_attention_all_gather_metadata.hpp"
#include "ring_attention_prefetch_utils.hpp"
#include <cstdint>
#include <utility>

using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
enum CompileTimeArg : uint32_t {
    kMyChipId,
    kCbOutputId,
    kPacketSizeInPages,
    kInputTensorPageSize,
    kNumTargetsForwardDirection,
    kNumTargetsBackwardDirection,
    kTopology,
    kContigPagesAdvanced,
    kNumInputs,
    kDirection,
    kFuseOp,
    kHasMetadata,
    kNumLinks,
    kSplitForwardingEnabled,
    kPartialReadinessEnabled,
    kOutputBankOwnedSchedule,
    kNumDramBanks,
    kPrefetchPackets,
    kNumFixedCompileTimeArgs,
};

constexpr uint32_t my_chip_id = get_compile_time_arg_val(kMyChipId);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(kCbOutputId);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(kPacketSizeInPages);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(kInputTensorPageSize);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(kNumTargetsForwardDirection);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(kNumTargetsBackwardDirection);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(kTopology));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(kContigPagesAdvanced);
constexpr uint32_t num_inputs = get_compile_time_arg_val(kNumInputs);
constexpr bool direction = get_compile_time_arg_val(kDirection);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(kFuseOp);
constexpr bool has_metadata = get_compile_time_arg_val(kHasMetadata);
constexpr uint32_t num_links = get_compile_time_arg_val(kNumLinks);
// The parent fused op owns this protocol decision and passes the same flag to both
// all-gather directions and its receiver, keeping producer and consumer in sync.
constexpr bool split_forwarding_enabled = get_compile_time_arg_val(kSplitForwardingEnabled);
constexpr bool partial_readiness_enabled = get_compile_time_arg_val(kPartialReadinessEnabled);
constexpr bool output_bank_owned_schedule = get_compile_time_arg_val(kOutputBankOwnedSchedule);
constexpr uint32_t num_dram_banks = get_compile_time_arg_val(kNumDramBanks);

static_assert(!partial_readiness_enabled || num_inputs == 1, "partial readiness requires one gathered input");
static_assert(
    !partial_readiness_enabled || !split_forwarding_enabled,
    "partial readiness and split forwarding are mutually exclusive");
static_assert(
    !partial_readiness_enabled || (num_targets_forward_direction > 0 && num_targets_backward_direction > 0),
    "partial readiness requires both ring directions");

// Prefetch: batch multiple packets of DRAM reads before a single barrier.
// This keeps more reads in flight across interleaved DRAM banks, hiding latency.
// CB depth must be >= 2 * prefetch_packets * packet_size_in_pages (see program_factory cb_num_pages).
constexpr uint32_t prefetch_packets = get_compile_time_arg_val(kPrefetchPackets);

template <bool physically_contiguous, typename Accessor>
FORCE_INLINE void prefetch_bank_owned_slices(
    const Noc& noc,
    CircularBuffer& cb_output,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    uint32_t accessor_page_base,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    ring_attention_all_gather::BankOwnedPacketSchedule<num_dram_banks> schedule(
        output_page_base, valid_pages, first_bank, bank_stride, packet_size_in_pages);
    const auto next_packet = [&](uint32_t& pages_to_read) {
        uint32_t first_page_offset = 0;
        schedule.next_packet(first_page_offset, pages_to_read);
        return accessor_page_base + first_page_offset;
    };
    if constexpr (physically_contiguous) {
        prefetch_batch_read_physically_contiguous_packets<
            input_tensor_page_size,
            packet_size_in_pages,
            prefetch_packets>(
            noc, cb_output, schedule.packets_remaining, cb_fifo_limit, cb_fifo_size, accessor, next_packet);
    } else {
        prefetch_batch_read_packets<input_tensor_page_size, packet_size_in_pages, prefetch_packets>(
            noc,
            cb_output,
            schedule.packets_remaining,
            cb_fifo_limit,
            cb_fifo_size,
            accessor,
            next_packet,
            [](uint32_t first_page_id, uint32_t page) { return first_page_id + page * num_dram_banks; });
    }
}

void kernel_main() {
    constexpr auto inputs_args = make_tensor_accessor_args_tuple<num_inputs, kNumFixedCompileTimeArgs + num_inputs>();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<
        num_inputs,
        std::get<num_inputs - 1>(inputs_args).next_compile_time_args_offset()>();
    constexpr uint32_t kMetaArgsOffset = has_metadata
                                             ? std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset()
                                             : (kNumFixedCompileTimeArgs + num_inputs);
    constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
    constexpr uint32_t kKvMetaArgsOffset = has_metadata ? meta_args.next_compile_time_args_offset() : kMetaArgsOffset;
    constexpr auto kv_meta_args = TensorAccessorArgs<kKvMetaArgsOffset>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // KEEP IN SYNC with kReaderRuntimeArgHeaderCount/kTensorDescriptorFieldCount and field offsets in
    // ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp. Fused consumers patch
    // selected fields through that shared host-side layout.
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
    // Nonzero only for a single-slot gather, where it skips to the selected input slot.
    // The slice is always emitted into output slot 0, whatever the output batch size.
    std::array<uint32_t, num_inputs> input_batch_base;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_base[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        // valid_pages_per_batch_head (slot 6): clamp the gather to the logical_n-valid slab prefix so
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

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    // Read the local slice into the packet CB before sending it over Fabric.
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
        const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
        if constexpr (output_bank_owned_schedule) {
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                const uint32_t input_page_base = input_batch_base[input_idx] + bh_idx * input_pages_per_batch_head;
                const uint32_t output_page_base =
                    bh_idx * output_pages_per_batch_head + my_chip_id * input_pages_per_batch_head;
                if constexpr (partial_readiness_enabled) {
                    const uint32_t first_pages = ring_attention_all_gather::midpoint_prefix_pages(
                        input_valid_pages[input_idx], input_tensor_Wt[input_idx]);
                    prefetch_bank_owned_slices<false>(
                        noc_obj,
                        cb_output,
                        cb_fifo_limit,
                        cb_fifo_size,
                        input_tensor_addrgens[input_idx],
                        input_page_base,
                        output_page_base,
                        first_pages,
                        worker_link[input_idx],
                        num_links);
                    prefetch_bank_owned_slices<false>(
                        noc_obj,
                        cb_output,
                        cb_fifo_limit,
                        cb_fifo_size,
                        input_tensor_addrgens[input_idx],
                        input_page_base + first_pages,
                        output_page_base + first_pages,
                        input_valid_pages[input_idx] - first_pages,
                        worker_link[input_idx],
                        num_links);
                } else {
                    prefetch_bank_owned_slices<false>(
                        noc_obj,
                        cb_output,
                        cb_fifo_limit,
                        cb_fifo_size,
                        input_tensor_addrgens[input_idx],
                        input_page_base,
                        output_page_base,
                        input_valid_pages[input_idx],
                        worker_link[input_idx],
                        num_links);
                }
            }
        } else {
            // For a single-slot gather this starts at the sliced batch slot; otherwise 0 (full batch).
            uint32_t input_page_base = input_batch_base[input_idx];
            uint32_t tiles_read = input_tile_id_start[input_idx];
            uint32_t tiles_to_read = input_tile_id_end[input_idx];
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                prefetch_batch_read_tiles<
                    input_tensor_page_size,
                    packet_size_in_pages,
                    prefetch_packets,
                    contig_pages_advanced>(
                    noc_obj,
                    cb_output,
                    tiles_read,
                    tiles_to_read,
                    cb_fifo_limit,
                    cb_fifo_size,
                    input_tensor_addrgens[input_idx],
                    [&](uint32_t tr) { return input_page_base + tr; });
                tiles_read = input_tile_id_start[input_idx];
                tiles_to_read = input_tile_id_end[input_idx];
                input_page_base += input_pages_per_batch_head;
            }
        }
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
        const uint32_t next_slice = slices_received + 1;
        if constexpr (partial_readiness_enabled) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), next_slice * 2 - 1);
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), next_slice);
        }

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if constexpr (direction == 1) {
            sender_chip_id = my_chip_id + next_slice;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - next_slice;
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }

        if constexpr (fuse_op) {
            // In partial mode this first signal exposes the completed prefix of the shard.
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
        if constexpr (partial_readiness_enabled) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), next_slice * 2);
            if constexpr (fuse_op) {
                op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
            }
        }
        slices_received = next_slice;
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
                const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
                const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
                if constexpr (output_bank_owned_schedule) {
                    // Split the diametric slice by DRAM-bank ownership. Both directions together still
                    // cover every bank, while each direction forwards roughly half the traffic.
                    const uint32_t split_factor = is_split_forwarded_slice ? 2 : 1;
                    const uint32_t first_bank =
                        worker_link[input_idx] + (is_split_forwarded_slice ? direction * num_links : 0);
                    const uint32_t bank_stride = num_links * split_factor;
                    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                        const uint32_t output_page_base =
                            bh_idx * output_pages_per_batch_head + actual_sender_chip_id * input_pages_per_batch_head;
                        if constexpr (partial_readiness_enabled) {
                            const uint32_t first_pages = ring_attention_all_gather::midpoint_prefix_pages(
                                input_valid_pages[input_idx], input_tensor_Wt[input_idx]);
                            prefetch_bank_owned_slices<true>(
                                noc_obj,
                                cb_output,
                                cb_fifo_limit,
                                cb_fifo_size,
                                output_tensor_addrgens[input_idx],
                                output_page_base,
                                output_page_base,
                                first_pages,
                                first_bank,
                                bank_stride);
                            prefetch_bank_owned_slices<true>(
                                noc_obj,
                                cb_output,
                                cb_fifo_limit,
                                cb_fifo_size,
                                output_tensor_addrgens[input_idx],
                                output_page_base + first_pages,
                                output_page_base + first_pages,
                                input_valid_pages[input_idx] - first_pages,
                                first_bank,
                                bank_stride);
                        } else {
                            prefetch_bank_owned_slices<true>(
                                noc_obj,
                                cb_output,
                                cb_fifo_limit,
                                cb_fifo_size,
                                output_tensor_addrgens[input_idx],
                                output_page_base,
                                output_page_base,
                                input_valid_pages[input_idx],
                                first_bank,
                                bank_stride);
                        }
                    }
                } else {
                    // Packet-aligned midpoint of this worker's page range (matches the writer).
                    const uint32_t total_pages = input_tile_id_end[input_idx] - input_tile_id_start[input_idx];
                    const uint32_t num_packets = (total_pages + packet_size_in_pages - 1) / packet_size_in_pages;
                    const uint32_t first_half_pages = (num_packets / 2) * packet_size_in_pages;
                    uint32_t relay_start = input_tile_id_start[input_idx];
                    uint32_t relay_end = input_tile_id_end[input_idx];
                    if (is_split_forwarded_slice) {
                        if constexpr (direction == 0) {
                            relay_end = relay_start + first_half_pages;
                        } else {
                            relay_start += first_half_pages;
                        }
                    }

                    uint32_t tiles_read = relay_start;
                    uint32_t tiles_to_read = relay_end;
                    uint32_t output_tile_id_start = 0;
                    const uint32_t slice_Wt = input_tensor_Wt[input_idx];
                    const uint32_t stride_Wt = output_tensor_Wt[input_idx];
                    uint32_t pages_read_in_row = relay_start % slice_Wt;
                    uint32_t row_offset = (relay_start / slice_Wt) * stride_Wt;
                    if (gather_dim == 3) {
                        output_tile_id_start = actual_sender_chip_id * slice_Wt;
                    } else {
                        output_tile_id_start = actual_sender_chip_id * input_pages_per_batch_head;
                    }
                    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                        prefetch_batch_read_tiles<
                            input_tensor_page_size,
                            packet_size_in_pages,
                            prefetch_packets,
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
                        output_tile_id_start += output_pages_per_batch_head;
                    }
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

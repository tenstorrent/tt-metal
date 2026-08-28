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

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(6));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(7);
constexpr uint32_t num_inputs = get_compile_time_arg_val(8);
constexpr bool direction = get_compile_time_arg_val(9);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(10);
constexpr bool has_metadata = get_compile_time_arg_val(11);
constexpr uint32_t cb_meta_id = get_compile_time_arg_val(12);
constexpr uint32_t num_links = get_compile_time_arg_val(13);
constexpr bool output_bank_owned_schedule = get_compile_time_arg_val(14);
constexpr uint32_t num_dram_banks = get_compile_time_arg_val(15);

// Prefetch: batch multiple packets of DRAM reads before a single barrier.
// This keeps more reads in flight across interleaved DRAM banks, hiding latency.
// CB depth must be >= 2 * PREFETCH_PACKETS * packet_size_in_pages (see program_factory cb_num_pages).
constexpr uint32_t PREFETCH_PACKETS = get_compile_time_arg_val(16);

template <bool physically_contiguous, typename Accessor>
FORCE_INLINE void prefetch_bank_owned_pages(
    const Noc& noc,
    CircularBuffer& cb_output,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    uint32_t source_page_base,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    ring_attention_all_gather::BankOwnedPacketSchedule<num_dram_banks> schedule(
        output_page_base, valid_pages, first_bank, bank_stride, packet_size_in_pages);
    while (schedule.packets_remaining > 0) {
        const uint32_t batch_packets = std::min(schedule.packets_remaining, PREFETCH_PACKETS);
        cb_output.reserve_back(batch_packets * packet_size_in_pages);
        uint32_t l1_write_addr = cb_output.get_write_ptr();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            if (l1_write_addr >= cb_fifo_limit) {
                l1_write_addr -= cb_fifo_size;
            }
            uint32_t first_page_offset = 0;
            uint32_t pages_to_read = 0;
            schedule.next_packet(packet_size_in_pages, first_page_offset, pages_to_read);
            if constexpr (physically_contiguous) {
                const uint64_t first_noc_addr =
                    accessor.get_noc_addr(source_page_base + first_page_offset, 0, noc.get_noc_id());
                noc.async_read(
                    tensor_accessor::Page(first_noc_addr, 0),
                    CoreLocalMem<uint8_t>(l1_write_addr),
                    pages_to_read * input_tensor_page_size,
                    {},
                    {});
            } else {
                for (uint32_t page = 0; page < pages_to_read; ++page) {
                    noc.async_read(
                        accessor,
                        CoreLocalMem<uint8_t>(l1_write_addr + page * input_tensor_page_size),
                        input_tensor_page_size,
                        {.page_id = source_page_base + first_page_offset + page * num_dram_banks},
                        {});
                }
            }
            l1_write_addr += packet_size_in_pages * input_tensor_page_size;
        }
        noc.async_read_barrier();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            cb_output.push_back(packet_size_in_pages);
        }
    }
}

void kernel_main() {
    constexpr uint32_t page_size_base_idx = 17;
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
    std::array<uint32_t, num_inputs> worker_index_in_link;
    std::array<uint32_t, num_inputs> workers_on_link;
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
        worker_index_in_link[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        workers_on_link[input_idx] = get_arg_val<uint32_t>(arg_idx++);
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
                const uint32_t first_bank = worker_link[input_idx] + worker_index_in_link[input_idx] * num_links;
                prefetch_bank_owned_pages<false>(
                    noc_obj,
                    cb_output,
                    cb_fifo_limit,
                    cb_fifo_size,
                    input_tensor_addrgens[input_idx],
                    input_page_base,
                    output_page_base,
                    input_valid_pages[input_idx],
                    first_bank,
                    num_links * workers_on_link[input_idx]);
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
                    PREFETCH_PACKETS,
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

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if constexpr (direction == 1) {
            sender_chip_id = my_chip_id + slices_received;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - slices_received;
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }

        if constexpr (fuse_op) {
            // Signal matmul to go
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
        // Direction == backward: Should I forward what I got from the left to my right?
        // In the linear case, if I have any targets to my right, always forward
        // In the ring case, if I have received on the left less than my targets on the right, forward
        // Direction == forward: Should I forward what I got from the right to my left?
        // In the linear case, if I have any targets to my left, always forward
        // In the ring case, if I have received on the right less than my targets on the left, forward
        if ((topology == Topology::Linear && writes_expected > 0) ||
            (topology == Topology::Ring && (slices_received < (writes_expected + 1)))) {
            for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
                const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
                const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
                if constexpr (output_bank_owned_schedule) {
                    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                        const uint32_t output_page_base =
                            bh_idx * output_pages_per_batch_head + actual_sender_chip_id * input_pages_per_batch_head;
                        const uint32_t first_bank =
                            worker_link[input_idx] + worker_index_in_link[input_idx] * num_links;
                        prefetch_bank_owned_pages<true>(
                            noc_obj,
                            cb_output,
                            cb_fifo_limit,
                            cb_fifo_size,
                            output_tensor_addrgens[input_idx],
                            output_page_base,
                            output_page_base,
                            input_valid_pages[input_idx],
                            first_bank,
                            num_links * workers_on_link[input_idx]);
                    }
                } else {
                    uint32_t tiles_read = input_tile_id_start[input_idx];
                    uint32_t tiles_to_read = input_tile_id_end[input_idx];
                    uint32_t output_tile_id_start = 0;
                    uint32_t pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                    uint32_t row_offset =
                        (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                    uint32_t slice_Wt = input_tensor_Wt[input_idx];
                    uint32_t stride_Wt = output_tensor_Wt[input_idx];
                    if (gather_dim == 3) {
                        output_tile_id_start = actual_sender_chip_id * input_tensor_Wt[input_idx];
                    } else {
                        output_tile_id_start = actual_sender_chip_id * input_pages_per_batch_head;
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
                        pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                        row_offset =
                            (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                        tiles_read = input_tile_id_start[input_idx];
                        tiles_to_read = input_tile_id_end[input_idx];
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

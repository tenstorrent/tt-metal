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
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
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
// Trace-safe metadata path: when set, the single-slot gather offset (input_batch_base) is recomputed
// on-device from slot_id = metadata[0] instead of taken from the (trace-frozen) runtime arg. cb_meta_id
// is a tiny L1 CB for the metadata page; the metadata accessor's compile args follow the output
// accessors. When has_metadata is false neither is emitted and this kernel is bit-identical.
constexpr bool has_metadata = get_compile_time_arg_val(11);
constexpr uint32_t cb_meta_id = get_compile_time_arg_val(12);

// Prefetch: batch multiple packets of DRAM reads before a single barrier.
// This keeps more reads in flight across interleaved DRAM banks, hiding latency.
// CB depth must be >= 2 * PREFETCH_PACKETS * packet_size_in_pages (see program_factory cb_num_pages).
constexpr uint32_t PREFETCH_PACKETS = 4;

// Batch-read tiles into the output CB with DRAM prefetch and FIFO wrapping.
// next_page_id is called once per tile-group read; it returns the source TensorAccessor
// page id and may perform per-tile side effects (e.g. row-stride tracking).
//
// Manual ring-buffer wrapping: batched reads may write across the FIFO boundary, which
// bypasses the CB's normal contiguous-write assumption (see cb_push_back). This is safe because
// reads target raw L1 addresses via CoreLocalMem and cb_push_back is called per-packet
// (packet_size_in_pages always divides cb_num_pages evenly).
template <typename Accessor, typename PageIdFn>
FORCE_INLINE void prefetch_batch_read_tiles(
    const Noc& noc_obj,
    CircularBuffer& cb_output,
    uint32_t& tiles_read,
    uint32_t tiles_to_read,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    PageIdFn&& next_page_id) {
    constexpr uint32_t payload_size_bytes = input_tensor_page_size * contig_pages_advanced;
    while (tiles_read < tiles_to_read) {
        uint32_t remaining_tiles = tiles_to_read - tiles_read;
        uint32_t remaining_packets = (remaining_tiles + packet_size_in_pages - 1) / packet_size_in_pages;
        uint32_t batch_packets = std::min(remaining_packets, PREFETCH_PACKETS);
        uint32_t batch_pages = batch_packets * packet_size_in_pages;

        cb_output.reserve_back(batch_pages);
        uint32_t l1_write_addr = cb_output.get_write_ptr();

        for (uint32_t p = 0; p < batch_packets; p++) {
            uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
            for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                if (l1_write_addr >= cb_fifo_limit) {
                    l1_write_addr -= cb_fifo_size;
                }
                noc_obj.async_read(
                    accessor,
                    CoreLocalMem<uint8_t>(l1_write_addr),
                    input_tensor_page_size,
                    {.page_id = next_page_id(tiles_read)},
                    {});
                l1_write_addr += payload_size_bytes;
                tiles_read += contig_pages_advanced;
            }
            l1_write_addr += (packet_size_in_pages - num_pages_to_read) * input_tensor_page_size;
        }
        noc_obj.async_read_barrier();
        for (uint32_t p = 0; p < batch_packets; p++) {
            cb_output.push_back(packet_size_in_pages);
        }
    }
}

void kernel_main() {
    constexpr uint32_t page_size_base_idx = 13;
    constexpr auto inputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<
        num_inputs,
        std::get<num_inputs - 1>(inputs_args).next_compile_time_args_offset()>();
    // The metadata accessor's compile args follow the output accessors (metadata path only). When
    // has_metadata is false the metadata accessor is NOT emitted, so fall back to a VALID (but unused)
    // accessor offset -- the inputs-accessor start -- rather than 0: TensorAccessorArgs<> is instantiated
    // here unconditionally (it is not a dependent template), and offset 0 names my_chip_id, which fails
    // the accessor's internal static_assert. meta_args is only *used* inside `if constexpr(has_metadata)`.
    constexpr uint32_t kMetaArgsOffset = has_metadata
                                             ? std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset()
                                             : (page_size_base_idx + num_inputs);
    constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();

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
    // Phase-1 input page base: nonzero only for single-slot gather (skip to the sliced input slot).
    // The slice is always emitted into output slot 0, whatever the output batch size.
    std::array<uint32_t, num_inputs> input_batch_base;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_start[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_end[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_base[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        // valid_pages_per_batch_head: clamp the gather to the logical_n-valid slab prefix so only
        // kv_actual-sized data moves. Uniform across cores/devices, so producer/consumer page counts
        // and the ring slice protocol stay matched. Default (full input) leaves the range unchanged.
        const uint32_t valid_pages = get_arg_val<uint32_t>(arg_idx++);
        if (valid_pages < input_tile_id_end[input_idx]) {
            input_tile_id_end[input_idx] = valid_pages;
        }
    }

    auto inputs_tuple = make_tensor_accessor_tuple(inputs_args, arg_idx);
    arg_idx += num_inputs;
    auto input_tensor_addrgens = make_abstract_tensor_accessor_wrappers(inputs_tuple);
    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_tensor_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);

    // Trace-safe single-slot gather: recompute input_batch_base from slot_id = metadata[0] on-device,
    // so a captured trace replays across cache slots (the host runtime-arg input_batch_base, set for the
    // capture-time slot, would otherwise be frozen). input_batch_base = slot * num_heads * Ht * Wt,
    // matching ring_attention_all_gather_async_detail::input_batch_base_pages. The metadata DRAM address
    // is the next runtime arg (emitted before the optional signaler args).
    if constexpr (has_metadata) {
        constexpr uint32_t kMetadataReadBytes = 16;  // [slot_id, actual_start, actual_end, pad] uint32
        const uint32_t metadata_addr = get_arg_val<uint32_t>(arg_idx++);
        Noc meta_noc;
        CircularBuffer cb_meta(cb_meta_id);
        const auto s_meta = TensorAccessor(meta_args, metadata_addr);
        meta_noc.async_read(
            s_meta, CoreLocalMem<uint8_t>(cb_meta.get_write_ptr()), kMetadataReadBytes, {.page_id = 0}, {});
        meta_noc.async_read_barrier();
        CoreLocalMem<volatile uint32_t> meta(cb_meta.get_write_ptr());
        const uint32_t slot_id = meta[0];
        // chunk_local_tiles (per-device Q slab in tiles) arrives as the next runtime arg, used to
        // recompute the gather extent on-device (so the gather moves only the logical_n-valid prefix
        // even when the host logical_n is a placeholder -- the runner passes only the metadata tensor).
        const uint32_t chunk_local_tiles = get_arg_val<uint32_t>(arg_idx++);
        // (user, layer)-major KV-cache batch dim: cache_batch_idx = slot_id * kv_cache_num_layers +
        // kv_cache_layer_idx (mirrors the SDPA reader / update_padded_kv_cache). slot_id = metadata[0]
        // holds only the user slot; the per-layer factor arrives as the next two runtime args (appended
        // after chunk_local_tiles, before the optional signaler args -- must be read here so OpSignaler
        // consumes the correct arg_idx). Defaults (1, 0) reduce to slot_id, keeping callers bit-identical.
        const uint32_t kv_cache_num_layers = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_cache_layer_idx = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_actual = meta[1];  // metadata[1] = actual_start = kv_actual_isl (tile-aligned)
        // gather_valid_Ht = ceil(logical_n / chunk_global) * chunk_local_tiles, mirroring the host
        // compute_gather_valid_Ht. logical_nt = kv_actual/32 + chunk_global_tiles; chunk_global_tiles =
        // chunk_local_tiles * ring_size. TILE_HEIGHT = 32.
        const uint32_t chunk_global_tiles = chunk_local_tiles * ring_size;
        const uint32_t logical_nt_local = (kv_actual >> 5) + chunk_global_tiles;
        const uint32_t valid_slabs = (logical_nt_local + chunk_global_tiles - 1) / chunk_global_tiles;
        const uint32_t gather_valid_Ht = valid_slabs * chunk_local_tiles;
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            const uint32_t cache_batch_idx = slot_id * kv_cache_num_layers + kv_cache_layer_idx;
            input_batch_base[input_idx] = cache_batch_idx * input_batch_head_count[input_idx] *
                                          input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
            const uint32_t valid_Ht =
                gather_valid_Ht < input_tensor_Ht[input_idx] ? gather_valid_Ht : input_tensor_Ht[input_idx];
            const uint32_t valid_pages = valid_Ht * input_tensor_Wt[input_idx];
            if (valid_pages < input_tile_id_end[input_idx]) {
                input_tile_id_end[input_idx] = valid_pages;
            }
        }
    }

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    // Push out our local slice
    // For a single-slot gather this starts at the sliced batch slot; otherwise 0 (full batch).
    uint32_t output_tile_id_start = 0;
    // Read local slice to our buffers, before sending them over
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        output_tile_id_start = input_batch_base[input_idx];
        uint32_t tiles_read = input_tile_id_start[input_idx];
        uint32_t tiles_to_read = input_tile_id_end[input_idx];
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
            prefetch_batch_read_tiles(
                noc_obj,
                cb_output,
                tiles_read,
                tiles_to_read,
                cb_fifo_limit,
                cb_fifo_size,
                input_tensor_addrgens[input_idx],
                [&](uint32_t tr) { return output_tile_id_start + tr; });
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
                    output_tile_id_start =
                        actual_sender_chip_id * input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                    prefetch_batch_read_tiles(
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
                    output_tile_id_start += output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
                }
            }
        }
    }
    // Device 2.0 migration: legacy primitive retained, out_ready_sem is a GlobalSemaphore address.
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}

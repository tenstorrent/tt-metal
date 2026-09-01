// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ring_attention_all_gather_metadata.hpp"
#include "ring_attention_prefetch_utils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

constexpr uint32_t my_ring_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(3);
constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_inputs = get_compile_time_arg_val(5);
constexpr uint32_t meta_cb_id = get_compile_time_arg_val(6);
constexpr uint32_t page_size_base_idx = 7;
constexpr uint32_t prefetch_packets = 4;

void kernel_main() {
    constexpr auto input_accessor_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();
    // Appended after the per-input accessors so existing compile-arg indices are untouched. When the
    // flag is clear the accessor still has to name a VALID accessor offset -- TensorAccessorArgs<> is
    // instantiated unconditionally and static_asserts on a non-accessor arg -- so fall back to the first
    // input's.
    // The tuple itself has no offset accessor; the last element's next offset is where the
    // accessors end.
    constexpr uint32_t halo_meta_flag_idx =
        std::get<num_inputs - 1>(input_accessor_args).next_compile_time_args_offset();
    constexpr bool has_halo_metadata = get_compile_time_arg_val(halo_meta_flag_idx) == 1;
    constexpr uint32_t slot_meta_args_offset =
        has_halo_metadata ? halo_meta_flag_idx + 1 : page_size_base_idx + num_inputs;
    constexpr auto slot_meta_args = TensorAccessorArgs<slot_meta_args_offset>();
    constexpr uint32_t kv_meta_args_offset =
        has_halo_metadata ? slot_meta_args.next_compile_time_args_offset() : slot_meta_args_offset;
    constexpr auto kv_meta_args = TensorAccessorArgs<kv_meta_args_offset>();

    uint32_t arg_idx = 0;
    const size_t incoming_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_stride_pages;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_start;
    std::array<uint32_t, num_inputs> input_tile_end;
    std::array<uint32_t, num_inputs> input_batch_base;
    for (uint32_t input = 0; input < num_inputs; ++input) {
        input_stride_pages[input] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_start[input] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_end[input] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_base[input] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Trace-safe metadata path. The halo's source group is linear in the chunk index, so on the scalar
    // path the host rewrites these page ranges every dispatch; a replayed trace never runs that rewrite
    // and would keep reading the capturing chunk's tail. Recompute the shift here instead. The block sits
    // after the per-input descriptors so the host relocation's field offsets stay put.
    if constexpr (has_halo_metadata) {
        const uint32_t slot_id_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_cache_num_layers = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_cache_layer_idx = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t kv_actual_isl_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t q_local_tile_rows = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t halo_tile_rows = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t source_device = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t baked_start_Ht = get_arg_val<uint32_t>(arg_idx++);
        Noc meta_noc;
        CircularBuffer cb_meta(meta_cb_id);
        const uint32_t slot_id =
            trace_metadata::read_metadata_scalar_u32(meta_noc, slot_meta_args, slot_id_addr, cb_meta.get_write_ptr());
        const uint32_t kv_cache_batch_idx = slot_id * kv_cache_num_layers + kv_cache_layer_idx;
        for (uint32_t input = 0; input < num_inputs; ++input) {
            input_batch_base[input] = kv_cache_batch_idx * input_batch_head_count[input] * input_stride_pages[input];
        }
        const uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
            meta_noc, kv_meta_args, kv_actual_isl_addr, cb_meta.get_write_ptr());
        const uint32_t tail_start_Ht = ring_attention_all_gather::compute_halo_tail_start_Ht(
            kv_actual_isl, q_local_tile_rows, ring_size, halo_tile_rows, source_device);
        for (uint32_t input = 0; input < num_inputs; ++input) {
            const uint32_t input_Wt = get_arg_val<uint32_t>(arg_idx++);
            ring_attention_all_gather::relocate_halo_range(
                tail_start_Ht * input_Wt, baked_start_Ht * input_Wt, input_tile_start[input], input_tile_end[input]);
        }
    }

    auto input_accessors_tuple = make_tensor_accessor_tuple(input_accessor_args, arg_idx);
    arg_idx += num_inputs;
    auto input_accessors = make_abstract_tensor_accessor_wrappers(input_accessors_tuple);

    OpSignaler op_signaler(arg_idx);

    Noc noc;
    CircularBuffer cb_output(cb_output_id);
    const uint32_t cb_fifo_limit = get_local_cb_interface(cb_output_id).fifo_limit;
    const uint32_t cb_fifo_size = get_local_cb_interface(cb_output_id).fifo_size;
    for (uint32_t input = 0; input < num_inputs; ++input) {
        for (uint32_t bh = 0; bh < input_batch_head_count[input]; ++bh) {
            uint32_t tiles_read = input_tile_start[input];
            prefetch_batch_read_tiles<input_page_size, packet_size_in_pages, prefetch_packets, 1>(
                noc,
                cb_output,
                tiles_read,
                input_tile_end[input],
                cb_fifo_limit,
                cb_fifo_size,
                input_accessors[input],
                [&](uint32_t tile) { return input_batch_base[input] + bh * input_stride_pages[input] + tile; });
        }
    }

    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(incoming_ready_sem), 1);
    constexpr uint32_t predecessor_ring_id = my_ring_id == 0 ? ring_size - 1 : my_ring_id - 1;
    op_signaler.synchronize_workers_and_signal_op(predecessor_ring_id);
    // Consume exactly one arrival. The wrapping atomic decrement preserves a concurrent remote
    // increment for the next exchange; a read-modify-write could lose it.
    noc_semaphore_inc(get_noc_addr(incoming_ready_sem), static_cast<uint32_t>(-1));
    noc_async_atomic_barrier();
}

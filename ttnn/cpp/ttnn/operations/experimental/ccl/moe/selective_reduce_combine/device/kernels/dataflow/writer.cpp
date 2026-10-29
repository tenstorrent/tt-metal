// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

#ifndef LOCAL_COMBINE
#include "combine_fabric.hpp"
#endif

// packet size bytes 4352
namespace detail {

// output is [select_experts_k ,tokens, hidden]
template <uint32_t TokensPerDevice>
inline uint32_t get_output_page_idx(const uint32_t t, const uint32_t k) {
    uint32_t t_idx = t % TokensPerDevice;
    return k * TokensPerDevice + t_idx;
}

template <bool Enable = false>
struct DoubleBuffer {
private:
    uint32_t idx;

public:
    DoubleBuffer(
        const Noc& /*noc*/,
        const uint32_t /*compute_cores_per_combine_core*/,
        const uint32_t /*sync_semaphore_addr*/,
        size_t& /*rt_arg_count*/) :
        idx(0) {};

    auto& operator++() {
        ++idx;
        return *this;
    }

    auto operator*() { return idx; }
};

template <>
struct DoubleBuffer<true> {
private:
    const Noc& noc;
    const uint32_t compute_cores_per_combine_core;
    const uint32_t sync_semaphore_addr;
    volatile tt_l1_ptr uint32_t* core_coords_ptr;

    bool idx;

public:
    // Runtime args: [x0, y0, x1, y1, ...] for each of the compute_cores_per_combine_core cores.
    DoubleBuffer(
        const Noc& noc,
        const uint32_t compute_cores_per_combine_core,
        const uint32_t sync_semaphore_addr,
        size_t& rt_arg_count) :
        noc(noc),
        compute_cores_per_combine_core(compute_cores_per_combine_core),
        sync_semaphore_addr(sync_semaphore_addr),
        core_coords_ptr(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(rt_arg_count))),
        idx(false) {
        rt_arg_count += 2 * compute_cores_per_combine_core;
    };

    DoubleBuffer& operator++() {
        noc.async_writes_flushed();

        for (uint32_t c = 0; c < compute_cores_per_combine_core; ++c) {
            const uint64_t sem_noc_addr = safe_get_noc_addr(
                core_coords_ptr[2 * c],
                core_coords_ptr[2 * c + 1],
                sync_semaphore_addr,
                /*noc_id=*/1);
            // Device 2.0 migration: legacy primitive retained: sync_semaphore_addr is a
            // fabric-mux sync address.
            noc_semaphore_inc</*posted=*/true>(sem_noc_addr, 1, /*noc_id=*/1);
        }
        idx = !idx;
        return *this;
    }

    auto operator*() { return idx; }
};

}  // namespace detail

void kernel_main() {
    DeviceZoneScopedN("Combine-writer");

    constexpr uint32_t dense_token_maps_cb_id = get_named_compile_time_arg_val("dense_token_maps_cb_id");
    constexpr uint32_t token_counts_cb_id = get_named_compile_time_arg_val("token_counts_cb_id");
    constexpr uint32_t data_cb_id = get_named_compile_time_arg_val("data_cb_id");
    constexpr uint32_t token_activations_cb_id = get_named_compile_time_arg_val("token_activations_cb_id");
    constexpr uint32_t activations_stride_elm = get_named_compile_time_arg_val("activations_stride_elm");
    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val("num_token_parallel_cores");
    constexpr uint32_t num_data_parallel_cores = get_named_compile_time_arg_val("num_data_parallel_cores");
    constexpr bool use_init_semaphore = get_named_compile_time_arg_val("use_init_semaphore") == 1;
    constexpr uint32_t num_local_experts = get_named_compile_time_arg_val("num_local_experts");
    constexpr uint32_t global_num_tokens = get_named_compile_time_arg_val("global_num_tokens");  // global token size
    constexpr uint32_t source_token_segment_buffer_size_bytes =
        get_named_compile_time_arg_val("source_token_segment_buffer_size_bytes");
    constexpr uint32_t source_block_size_bytes = get_named_compile_time_arg_val("source_expert_block_size_bytes");
    constexpr uint32_t dense_token_maps_stride_elm = get_named_compile_time_arg_val("dense_token_maps_stride_elm");
    constexpr uint32_t compute_sync_semaphore_id = get_named_compile_time_arg_val("compute_sync_semaphore_id");
    constexpr bool double_buffer_source = get_named_compile_time_arg_val("double_buffer_source") == 1;

#ifdef LOCAL_COMBINE
    // Single-device local combine: tokens_per_device = global_num_tokens (1 device).
    constexpr uint32_t tokens_per_device = global_num_tokens;
    constexpr auto output_ta_args = TensorAccessorArgs<0>();
#else
    constexpr uint32_t tokens_per_device = global_num_tokens / combine_fabric::replicate_group_devices;
    constexpr auto output_ta_args = TensorAccessorArgs<0>();
#endif

    size_t rt_arg_count = 0;
    const auto output_base_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto source_token_segment_size_bytes = get_arg_val<uint32_t>(rt_arg_count++);
    const auto dest_token_segment_offset_bytes = get_arg_val<uint32_t>(rt_arg_count++);
    // Device 2.0 migration: legacy primitive retained: init_semaphore_addr is a GlobalSemaphore address.
    const auto init_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto global_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const bool is_init_sync_core = get_arg_val<uint32_t>(rt_arg_count++);

    Noc noc1_obj(1);
    CircularBuffer cb_token_counts(token_counts_cb_id);
    CircularBuffer cb_data(data_cb_id);
    CircularBuffer cb_dense_token_maps(dense_token_maps_cb_id);
    CircularBuffer cb_token_activations(token_activations_cb_id);
    Semaphore<> compute_sync_sem(compute_sync_semaphore_id);

    const auto compute_sync_semaphore_addr = get_semaphore(compute_sync_semaphore_id);

    // Double-buffered source only: number of compute cores feeding this combine core. It varies per
    // combine core because a compute core's width slice may straddle combine columns.
    const uint32_t compute_cores_per_combine_core = double_buffer_source ? get_arg_val<uint32_t>(rt_arg_count++) : 0;

    // rt_arg_count is incremented
    detail::DoubleBuffer<double_buffer_source> db(
        noc1_obj, compute_cores_per_combine_core, compute_sync_semaphore_addr, rt_arg_count);

    const auto output_addrgen = TensorAccessor(output_ta_args, output_base_addr);

#ifndef LOCAL_COMBINE
    // Fabric Mux V2 clients, opened blocking: eager staging would only hide the mux setup, and the first send waits on
    // compute, which takes longer.
    combine_fabric::Connections fabric(rt_arg_count);
    fabric.open();
    if (is_init_sync_core && use_init_semaphore) {
        fabric.send_init(init_semaphore_addr);
    }
#endif

    cb_token_counts.wait_front(1);
    const uint32_t token_counts_l1_addr = cb_token_counts.get_write_ptr();
    auto* token_counts_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_counts_l1_addr);
    uint32_t token_split_offsets[num_local_experts];
    uint32_t token_split_counts[num_local_experts];
    uint32_t token_activation_offsets[num_local_experts];
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        token_split_offsets[e] = token_counts_l1_ptr[num_local_experts + e];
        token_split_counts[e] = token_counts_l1_ptr[num_local_experts + num_local_experts + e];
        token_activation_offsets[e] = token_counts_l1_ptr[num_local_experts + 2 * num_local_experts + e];
    }
    cb_token_counts.pop_front(1);

    cb_data.reserve_back(1);
    const uint32_t src_data_l1_base_addr = cb_data.get_write_ptr();

    cb_dense_token_maps.wait_front(num_local_experts);
    const uint32_t dense_token_maps_l1_addr = cb_dense_token_maps.get_write_ptr();
    auto* dense_token_maps_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense_token_maps_l1_addr);

    cb_token_activations.wait_front(1);
    const uint32_t token_activations_l1_addr = cb_token_activations.get_write_ptr();
    auto* token_activations_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_activations_l1_addr);

#ifndef LOCAL_COMBINE
    // Local combine needs no cross-device start: the compute sync semaphore and the CB waits order it.
    if constexpr (use_init_semaphore) {
        fabric.wait_init(
            noc1_obj, init_semaphore_addr, is_init_sync_core, num_token_parallel_cores * num_data_parallel_cores);
    }
#endif

    uint32_t compute_sync_semaphore_val = compute_cores_per_combine_core;
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        auto* expert_token_activations_ptr =
            token_activations_l1_ptr + token_activation_offsets[e] * activations_stride_elm;

        compute_sync_sem.wait(compute_sync_semaphore_val);

        for (uint32_t dt = 0; dt < token_split_counts[e]; ++dt) {
            const uint32_t st = dense_token_maps_l1_ptr
                [(e * (global_num_tokens + 1) + token_split_offsets[e] + dt) * dense_token_maps_stride_elm];
            uint32_t guard = 0;
            while (expert_token_activations_ptr[0] != st) {
                expert_token_activations_ptr += activations_stride_elm;
                ASSERT(guard++ < global_num_tokens);
            }
            const uint32_t k = expert_token_activations_ptr[1 + e];

            // figure out output page index, noc address.
            const uint32_t output_page_idx = detail::get_output_page_idx<tokens_per_device>(st, k);

            const uint32_t src_data_l1_addr =
                src_data_l1_base_addr + *db * source_block_size_bytes + dt * source_token_segment_buffer_size_bytes;

#ifdef LOCAL_COMBINE
            // Local combine: all writes are local NOC writes (single device).
            noc1_obj.async_write(
                CoreLocalMem<uint8_t>(src_data_l1_addr),
                output_addrgen,
                source_token_segment_size_bytes,
                {},
                {.page_id = output_page_idx, .offset_bytes = dest_token_segment_offset_bytes});
            noc1_obj.async_writes_flushed();
#else
            const uint32_t dest_device_idx = combine_fabric::device_of_token<tokens_per_device>(st);
            if (dest_device_idx == combine_fabric::linearized_mesh_coord) {
                noc1_obj.async_write(
                    CoreLocalMem<uint8_t>(src_data_l1_addr),
                    output_addrgen,
                    source_token_segment_size_bytes,
                    {},
                    {.page_id = output_page_idx, .offset_bytes = dest_token_segment_offset_bytes});
                noc1_obj.async_writes_flushed();
            } else {
                fabric.send(
                    output_addrgen,
                    dest_device_idx,
                    src_data_l1_addr,
                    output_page_idx,
                    source_token_segment_size_bytes,
                    dest_token_segment_offset_bytes);
            }
#endif
        }
        compute_sync_semaphore_val += compute_cores_per_combine_core;
        ++db;
    }

    compute_sync_sem.set(0);

    cb_dense_token_maps.pop_front(num_local_experts);
    cb_token_activations.pop_front(1);
    cb_data.push_back(1);

    noc1_obj.async_write_barrier();
#ifdef LOCAL_COMBINE
    // Blackhole atomics are non-posted: without their acks the next kernel's atomic barrier on this NoC never completes
    noc1_obj.async_atomic_barrier();
#else
    fabric.finish(noc1_obj, global_semaphore_addr);
#endif
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

namespace detail {

template <uint32_t NumLocalExperts, uint32_t NumTokenParallelCores>
void token_work_split_even(
    const uint32_t token_parallel_core_id,
    volatile tt_l1_ptr uint32_t* dense_token_counts_ptr,
    uint32_t* token_split_counts,
    uint32_t* token_split_offsets) {
    for (uint32_t e = 0; e < NumLocalExperts; ++e) {
        token_split_counts[e] = 0;
        token_split_offsets[e] = 0;
        const uint32_t chunk = dense_token_counts_ptr[e] / NumTokenParallelCores;
        const uint32_t rem = dense_token_counts_ptr[e] % NumTokenParallelCores;
        for (uint32_t c = 0; c < NumTokenParallelCores; ++c) {
            const auto count = (c < rem) ? chunk + 1 : chunk;

            if (c == token_parallel_core_id) {
                token_split_counts[e] = count;
                break;
            }
            token_split_offsets[e] += count;
        }
    }
}

template <uint32_t NumLocalExperts, uint32_t ActivationStride, uint32_t MapStride, uint32_t GlobalNumTokens>
void get_token_activation_offsets(
    const uint32_t* token_split_offsets,
    const uint32_t* token_split_counts,
    volatile tt_l1_ptr uint32_t* dense_token_maps_ptr,
    volatile tt_l1_ptr uint32_t* token_activations_ptr,
    uint32_t* token_activation_offsets) {
    for (uint32_t e = 0; e < NumLocalExperts; ++e) {
        if (token_split_counts[e] == 0) {
            continue;
        }

        const auto token_split_offset = token_split_offsets[e];
        auto* expert_token_activations_ptr = token_activations_ptr + token_split_offset * ActivationStride;
        const auto st_start = dense_token_maps_ptr[(e * (GlobalNumTokens + 1) + token_split_offset) * MapStride];

        for (uint32_t t = token_split_offset; t < GlobalNumTokens; ++t) {
            if (expert_token_activations_ptr[0] == st_start) {
                token_activation_offsets[e] = t;
                break;
            }
            expert_token_activations_ptr += ActivationStride;
            ASSERT(t != GlobalNumTokens - 1);
        }
    }
}

}  // namespace detail
void kernel_main() {
    constexpr uint32_t dense_token_maps_cb_id = get_named_compile_time_arg_val("dense_token_maps_cb_id");
    constexpr uint32_t token_counts_cb_id = get_named_compile_time_arg_val("token_counts_cb_id");
    constexpr uint32_t token_activations_cb_id = get_named_compile_time_arg_val("token_activations_cb_id");
    constexpr uint32_t dense_token_maps_page_size_bytes =
        get_named_compile_time_arg_val("dense_token_maps_page_size_bytes");
    constexpr uint32_t token_counts_page_size_bytes = get_named_compile_time_arg_val("token_counts_page_size_bytes");
    constexpr uint32_t token_activations_page_size_bytes =
        get_named_compile_time_arg_val("token_activations_page_size_bytes");
    constexpr uint32_t aligned_token_activations_page_size_bytes =
        get_named_compile_time_arg_val("aligned_token_activations_page_size_bytes");
    constexpr uint32_t activations_stride_elm = get_named_compile_time_arg_val("activations_stride_elm");
    constexpr uint32_t dense_token_maps_stride_elm = get_named_compile_time_arg_val("dense_token_maps_stride_elm");
    constexpr uint32_t num_local_experts = get_named_compile_time_arg_val("num_local_experts");
    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val("num_token_parallel_cores");
    constexpr uint32_t num_data_parallel_cores = get_named_compile_time_arg_val("num_data_parallel_cores");
    constexpr uint32_t global_num_tokens = get_named_compile_time_arg_val("global_num_tokens");
    constexpr uint32_t select_experts_k = get_named_compile_time_arg_val("select_experts_k");
    constexpr uint32_t sync_semaphore_id = get_named_compile_time_arg_val("sync_semaphore_id");
    constexpr uint32_t noc_x_start = get_named_compile_time_arg_val("noc_x_start");
    constexpr uint32_t noc_y_start = get_named_compile_time_arg_val("noc_y_start");
    constexpr uint32_t noc_x_end = get_named_compile_time_arg_val("noc_x_end");
    constexpr uint32_t noc_y_end = get_named_compile_time_arg_val("noc_y_end");

    constexpr uint32_t aligned_activations_page_size = aligned_token_activations_page_size_bytes / sizeof(uint32_t);

    constexpr auto dense_token_maps_ta_args = TensorAccessorArgs<0>();
    constexpr auto dense_token_counts_ta_args =
        TensorAccessorArgs<dense_token_maps_ta_args.next_compile_time_args_offset()>();
    constexpr auto token_activations_ta_args =
        TensorAccessorArgs<dense_token_counts_ta_args.next_compile_time_args_offset()>();
    uint32_t arg_index = 0;
    const auto dense_token_maps_addr = get_arg_val<uint32_t>(arg_index++);
    const auto dense_token_counts_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_activations_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_parallel_core_id = get_arg_val<uint32_t>(arg_index++);
    const bool sync_core = get_arg_val<uint32_t>(arg_index++);

    const uint32_t sync_semaphore_addr = get_semaphore(sync_semaphore_id);

    const auto dense_token_maps_addrgen =
        TensorAccessor(dense_token_maps_ta_args, dense_token_maps_addr, dense_token_maps_page_size_bytes);
    const auto token_counts_addrgen =
        TensorAccessor(dense_token_counts_ta_args, dense_token_counts_addr, token_counts_page_size_bytes);
    const auto token_activations_addrgen =
        TensorAccessor(token_activations_ta_args, token_activations_addr, token_activations_page_size_bytes);

    // wait for metadata to be ready
    auto* sync_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_semaphore_addr);
    if (sync_core) {
        noc_semaphore_wait(sync_semaphore_ptr, 1);
        // swap start/end coordinates because this kernel is using NOC1
        const uint64_t semaphore_mc_addr =
            get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, sync_semaphore_addr, /*noc=*/1);
        noc_semaphore_set_multicast(
            sync_semaphore_addr,
            semaphore_mc_addr,
            num_token_parallel_cores * num_data_parallel_cores - 1,
            /*linked=*/false,
            /*noc=*/1);
        noc_async_writes_flushed(/*noc=*/1);

    } else {
        noc_semaphore_wait_min(sync_semaphore_ptr, 1);
    }

    // read dense token counts
    cb_reserve_back(token_counts_cb_id, 1);
    const uint32_t token_counts_l1_addr = get_read_ptr(token_counts_cb_id);
    const uint64_t token_counts_noc_addr = get_noc_addr(0, token_counts_addrgen, /*offset=*/0, /*noc=*/1);
    noc_async_read(token_counts_noc_addr, token_counts_l1_addr, token_counts_page_size_bytes, /*noc=*/1);
    noc_async_read_barrier(/*noc=*/1);

    // read activations
    cb_reserve_back(token_activations_cb_id, 1);
    const uint32_t token_activations_l1_addr = get_write_ptr(token_activations_cb_id);
    const uint64_t token_activations_noc_addr = get_noc_addr(0, token_activations_addrgen, /*offset=*/0, /*noc=*/1);
    noc_async_read(
        token_activations_noc_addr, token_activations_l1_addr, aligned_token_activations_page_size_bytes, /*noc=*/1);

    // split work
    auto* token_counts_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_counts_l1_addr);
    uint32_t token_split_offsets[num_local_experts];
    uint32_t token_split_counts[num_local_experts];
    detail::token_work_split_even<num_local_experts, num_token_parallel_cores>(
        token_parallel_core_id, token_counts_l1_ptr, token_split_counts, token_split_offsets);

    // read dense token maps
    cb_reserve_back(dense_token_maps_cb_id, num_local_experts);
    const uint32_t dense_token_maps_l1_addr = get_write_ptr(dense_token_maps_cb_id);
    for (uint32_t e = 0, l1_offset = 0, maps_page = 0; e < num_local_experts; ++e) {
        const uint64_t dense_token_maps_noc_addr =
            get_noc_addr(maps_page++, dense_token_maps_addrgen, /*offset=*/0, /*noc=*/1);
        noc_async_read(
            dense_token_maps_noc_addr,
            dense_token_maps_l1_addr + l1_offset,
            dense_token_maps_page_size_bytes,
            /*noc=*/1);
        l1_offset += dense_token_maps_page_size_bytes;
    }

    // wait for activations and dense token maps
    noc_async_read_barrier(/*noc=*/1);

    auto* dense_token_maps_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense_token_maps_l1_addr);
    auto* token_activations_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_activations_l1_addr);
    uint32_t token_activation_offsets[num_local_experts];
    detail::get_token_activation_offsets<
        num_local_experts,
        activations_stride_elm,
        dense_token_maps_stride_elm,
        global_num_tokens>(
        token_split_offsets,
        token_split_counts,
        dense_token_maps_l1_ptr,
        token_activations_ptr,
        token_activation_offsets);

    // stash the work split counts, offsets at the end of the token counts
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        token_counts_l1_ptr[num_local_experts + e] = token_split_offsets[e];
        token_counts_l1_ptr[num_local_experts + num_local_experts + e] = token_split_counts[e];
        token_counts_l1_ptr[num_local_experts + 2 * num_local_experts + e] = token_activation_offsets[e];
    }

    cb_push_back(token_counts_cb_id, 1);
    cb_push_back(dense_token_maps_cb_id, num_local_experts);
    cb_push_back(token_activations_cb_id, 1);

    noc_semaphore_set(sync_semaphore_ptr, 0);
}

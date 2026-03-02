// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

namespace detail {

inline uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }

// this algorithm matches the current implementation in MoE compute
template <uint32_t NumLocalExperts, uint32_t NumTokenParallelCores>
void token_work_split_simple(
    const uint32_t token_parallel_core_id,
    volatile tt_l1_ptr uint32_t* dense_token_counts_ptr,
    uint32_t* token_split_counts,
    uint32_t* token_split_offsets) {
    for (uint32_t e = 0; e < NumLocalExperts; ++e) {
        token_split_offsets[e] = 0;
        const uint32_t chunk = div_up(dense_token_counts_ptr[e], NumTokenParallelCores);
        const uint32_t rem = dense_token_counts_ptr[e] - (NumTokenParallelCores - 1) * chunk;

        for (uint32_t c = 0; c < NumTokenParallelCores; ++c) {
            const uint32_t count = (c == NumTokenParallelCores - 1) ? rem : chunk;

            if (c == token_parallel_core_id) {
                token_split_counts[e] = count;
                break;
            }

            token_split_offsets[e] += count;
        }
    }
}

// this algorithm more efficiently splits work, switch to it eventually
template <uint32_t NumLocalExperts, uint32_t NumTokenParallelCores>
[[maybe_unused]] void token_work_split_even(
    const uint32_t token_parallel_core_id,
    volatile tt_l1_ptr uint32_t* dense_token_counts_ptr,
    uint32_t* token_split_counts,
    uint32_t* token_split_offsets) {
    for (uint32_t e = 0; e < NumLocalExperts; ++e) {
        token_split_offsets[e] = 0;
        for (uint32_t c = 0; c < NumTokenParallelCores; ++c) {
            uint32_t count = dense_token_counts_ptr[e] / NumTokenParallelCores;
            if (c < dense_token_counts_ptr[e] % NumTokenParallelCores) {
                ++count;
            }

            if (c == token_parallel_core_id) {
                token_split_counts[e] = count;
                break;
            }
            token_split_offsets[e] += count;
        }
    }
}
}  // namespace detail
void kernel_main() {
    constexpr uint32_t dense_token_maps_cb_id = get_named_compile_time_arg_val("dense_token_maps_cb_id");
    constexpr uint32_t token_counts_cb_id = get_named_compile_time_arg_val("token_counts_cb_id");
    constexpr uint32_t dense_token_maps_page_size_bytes =
        get_named_compile_time_arg_val("dense_token_maps_page_size_bytes");
    constexpr uint32_t token_counts_page_size_bytes = get_named_compile_time_arg_val("token_counts_page_size_bytes");
    constexpr uint32_t num_local_experts = get_named_compile_time_arg_val("num_local_experts");
    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val("num_token_parallel_cores");
    constexpr uint32_t global_num_tokens = get_named_compile_time_arg_val("global_num_tokens");
    constexpr uint32_t select_experts_k = get_named_compile_time_arg_val("select_experts_k");
    constexpr uint32_t sync_semaphore_id = get_named_compile_time_arg_val("sync_semaphore_id");
    constexpr uint32_t noc_x_start = get_named_compile_time_arg_val("noc_x_start");
    constexpr uint32_t noc_y_start = get_named_compile_time_arg_val("noc_y_start");
    constexpr uint32_t noc_x_end = get_named_compile_time_arg_val("noc_x_end");
    constexpr uint32_t noc_y_end = get_named_compile_time_arg_val("noc_y_end");

    constexpr auto dense_token_maps_ta_args = TensorAccessorArgs<0>();
    constexpr auto dense_token_counts_ta_args = TensorAccessorArgs<1>();

    uint32_t arg_index = 0;
    const auto dense_token_maps_addr = get_arg_val<uint32_t>(arg_index++);
    const auto dense_token_counts_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_parallel_core_id = get_arg_val<uint32_t>(arg_index++);

    const auto sync_semaphore_addr = get_semaphore(sync_semaphore_id);

    const auto dense_token_maps_addrgen =
        TensorAccessor(dense_token_maps_ta_args, dense_token_maps_addr, dense_token_maps_page_size_bytes);
    const auto token_counts_addrgen =
        TensorAccessor(dense_token_counts_ta_args, dense_token_counts_addr, token_counts_page_size_bytes);

    // wait for metadata to be ready
    auto* sync_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_semaphore_addr);
    if (sync_core) {
        noc_semaphore_wait(sync_semaphore_ptr, 1);
        const uint64_t semaphore_mc_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, sync_semaphore_addr);
        noc_semaphore_set_multicast(
            sync_semaphore_addr, semaphore_mc_addr, num_token_parallel_cores * num_data_parallel_cores - 1);
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_wait(sync_semaphore_ptr, 1);
    }
    noc_semaphore_set(sync_semaphore_ptr, 0);

    // read dense token counts
    cb_reserve_back(token_counts_cb_id, 1);
    const uint32_t token_counts_l1_addr = get_read_ptr(token_counts_cb_id);
    const uint64_t token_counts_noc_addr = get_noc_addr(0, token_counts_addrgen);
    noc_async_read(token_counts_noc_addr, token_counts_l1_addr, token_counts_page_size_bytes);
    auto* token_counts_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_counts_l1_addr);
    noc_async_read_barrier();
    cb_push_back(token_counts_cb_id, 1);

    // read dense token maps
    cb_reserve_back(dense_token_maps_cb_id, 1);
    const uint32_t dense_token_maps_l1_addr = get_read_ptr(dense_token_maps_cb_id);
    const uint64_t dense_token_maps_noc_addr = get_noc_addr(0, dense_token_maps_addrgen);
    noc_async_read(dense_token_maps_noc_addr, dense_token_maps_l1_addr, dense_token_maps_page_size_bytes);

    // split work
    uint32_t token_split_offsets[num_local_experts];
    uint32_t token_split_counts[num_local_experts];
    detail::token_work_split_simple<num_local_experts, num_token_parallel_cores>(
        token_parallel_core_id, token_counts_l1_ptr, token_split_counts, token_split_offsets);

    // stash the work split counts, offsets at the end of the token counts
    auto* dense_token_maps_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense_token_maps_l1_addr);

    for (uint32_t e = 0; e < num_local_experts; ++e) {
        dense_token_maps_l1_ptr[num_local_experts * global_num_tokens + e] = token_split_offsets[e];
        dense_token_maps_l1_ptr[num_local_experts * global_num_tokens + num_local_experts + e] = token_split_counts[e];
    }
    noc_async_read_barrier();
    cb_push_back(dense_token_maps_cb_id, 1);
}

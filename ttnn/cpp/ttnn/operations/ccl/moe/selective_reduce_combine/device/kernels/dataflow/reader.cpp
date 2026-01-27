// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

inline void print_u32_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << (uint32_t)*ptr << " ";
        }
        DPRINT << ENDL();
    }
}

namespace detail {

template <uint32_t NumLocalExperts, uint32_t NumTokenParallelCores>
void token_work_split(
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
}//namespace detail
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

    constexpr auto dense_token_maps_ta_args = TensorAccessorArgs<0>();
    constexpr auto dense_token_counts_ta_args = TensorAccessorArgs<1>();

    uint32_t arg_index = 0;
    const auto dense_token_maps_addr = get_arg_val<uint32_t>(arg_index++);
    const auto dense_token_counts_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_parallel_core_id = get_arg_val<uint32_t>(arg_index++);

    const auto dense_token_maps_addrgen =
        TensorAccessor(dense_token_maps_ta_args, dense_token_maps_addr, dense_token_maps_page_size_bytes);
    const auto token_counts_addrgen =
        TensorAccessor(dense_token_counts_ta_args, dense_token_counts_addr, token_counts_page_size_bytes);

    // read dense token counts
    cb_reserve_back(token_counts_cb_id, 1);
    const uint32_t token_counts_l1_addr = get_read_ptr(token_counts_cb_id);
    const uint64_t token_counts_noc_addr = get_noc_addr(0, token_counts_addrgen);
    noc_async_read(token_counts_noc_addr, token_counts_l1_addr, token_counts_page_size_bytes);
    auto* token_counts_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_counts_l1_addr);
    noc_async_read_barrier();
    cb_push_back(token_counts_cb_id, 1);

    print_u32_pages(token_counts_l1_addr, num_local_experts, 1);

    // read dense token maps
    cb_reserve_back(dense_token_maps_cb_id, 1);
    const uint32_t dense_token_maps_l1_addr = get_read_ptr(dense_token_maps_cb_id);
    const uint64_t dense_token_maps_noc_addr = get_noc_addr(0, dense_token_maps_addrgen);
    noc_async_read(dense_token_maps_noc_addr, dense_token_maps_l1_addr, dense_token_maps_page_size_bytes);

    // split work
    uint32_t token_split_offsets[num_local_experts];
    uint32_t token_split_counts[num_local_experts];
    detail::token_work_split<num_local_experts, num_token_parallel_cores>(
        token_parallel_core_id, token_counts_l1_ptr, token_split_counts, token_split_offsets);

    // stash the work split counts, offsets at the end of the token counts
    auto* dense_token_maps_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense_token_maps_l1_addr);
    DPRINT << "OFFSETS: " << token_split_offsets[0] << " " << token_split_offsets[1] << " token_split_counts "
           << token_split_counts[0] << " " << token_split_counts[1] << "\n";

    for (uint32_t e = 0; e < num_local_experts; ++e) {
        dense_token_maps_l1_ptr[num_local_experts * global_num_tokens + e] = token_split_offsets[e];
        dense_token_maps_l1_ptr[num_local_experts * global_num_tokens + num_local_experts + e] = token_split_counts[e];
    }
    noc_async_read_barrier();
    cb_push_back(dense_token_maps_cb_id, 1);
}

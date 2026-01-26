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

template <uint32_t NumLocalExperts>
inline uint32_t get_max_token_count(volatile tt_l1_ptr uint32_t* counts_ptr) {
    uint32_t max = 0;
    for(uint32_t i =0; i < NumLocalExperts; ++i){
        if (counts_ptr[i] > max) {
            max = count_ptr[i];
        }
    }
    return max;
}

template <
    uint32_t GlobalNumTokens,
    uint32_t SelectExpertsK,
    uint32_t NumLocalExperts,
    uint32_t NumTokenParallelCores,
    uint32_t MetaDataEntrySize>
void token_work_split(
    volatile tt_l1_ptr uint32_t* dense_token_counts_ptr,
    volatile tt_l1_ptr uint32_t* dense_metadata_ptr,
    const uint32_t token_parallel_core_id,
    uint32_t& metadata_start_offset,
    uint32_t& token_end,
    int32_t* edt) {
    constexpr auto invalid = GlobalNumTokens + 1;
    const auto max_active_tokens = get_max_token_count<NumLocalExperts>(dense_token_counts_ptr);

    constexpr uint32_t token_core_block_size = GlobalNumTokens / NumTokenParallelCores;
    const int32_t core_token_index_start = token_parallel_core_id * token_core_block_size;
    const int32_t core_token_index_end = core_token_index_start + std::min(max_active_tokens, token_core_block_size);

    DPRINT << "max_active_tokens: " << max_active_tokens << " core_token_index_end: " << core_token_index_end << "\n";
    metadata_start_offset = invalid;
    token_end = 0;

    for (uint32_t e = 0; e < NumLocalExperts; ++e) {
        edt[e] = 0;
    }

    bool end = false;
    int32_t max_edt = 0;
    for (uint32_t dt = 0; dt < GlobalNumTokens; ++dt) {
        for (uint8_t e = 0; e < NumLocalExperts; ++e) {
            if (edt[e] == core_token_index_start) {
                if (metadata_start_offset == invalid) {
                    metadata_start_offset = dt;
                }
            }

            if (edt[e] == core_token_index_end) {
                token_end = dt + 1;
                end = true;
                break;
            }
            const auto k = dense_metadata_ptr[e + 1];
            if (k != SelectExpertsK + 1) {
                if (++edt[e] > max_edt) {
                    max_edt = edt[e];
                }
            }
        }
        if (end) {
            break;
        }
        dense_metadata_ptr += MetaDataEntrySize;
    }

    if (token_parallel_core_id == 0) {
        for (uint8_t e = 0; e < NumLocalExperts; ++e) {
            edt[e] = 0;
        }
    } else {
        for (uint8_t e = 0; e < NumLocalExperts; ++e) {
            edt[e] -= max_edt;
        }
    }
}

}//namespace detail
void kernel_main() {
    constexpr uint32_t dense_metadata_cb_id = get_named_compile_time_arg_val("metadata_cb_id");
    constexpr uint32_t token_counts_cb_id = get_named_compile_time_arg_val("token_counts_cb_id");
    constexpr uint32_t token_counts_buffer_size_bytes =
        get_named_compile_time_arg_val("token_counts_buffer_size_bytes");
    constexpr uint32_t num_local_experts = get_named_compile_time_arg_val("num_local_experts");
    constexpr uint32_t metadata_entry_size = get_named_compile_time_arg_val("metadata_entry_size");
    constexpr uint32_t metadata_entry_size_bytes = get_named_compile_time_arg_val("metadata_entry_size_bytes");
    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val("num_token_parallel_cores");
    constexpr uint32_t global_num_tokens = get_named_compile_time_arg_val("global_num_tokens");
    constexpr uint32_t select_experts_k = get_named_compile_time_arg_val("select_experts_k");

    constexpr uint32_t metadata_size_bytes = metadata_entry_size_bytes * global_num_tokens;

    constexpr auto metadata_ta_args = TensorAccessorArgs<0>();
    constexpr auto dense_token_counts_ta_args = TensorAccessorArgs<1>();

    uint32_t arg_index = 0;
    const auto dense_metadata_addr = get_arg_val<uint32_t>(arg_index++);
    const auto dense_token_counts_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_parallel_core_id = get_arg_val<uint32_t>(arg_index++);

    const auto metadata_addrgen = TensorAccessor(metadata_ta_args, dense_metadata_addr, metadata_size_bytes);
    const auto token_counts_addrgen =
        TensorAccessor(metadata_ta_args, dense_metadata_addr, token_counts_buffer_size_bytes);

    // read dense metadata
    cb_reserve_back(dense_metadata_cb_id, 1);
    const uint32_t metadata_cb_addr = get_read_ptr(dense_metadata_cb_id);
    const uint64_t metadata_noc_addr = get_noc_addr(0, metadata_addrgen);
    noc_async_read(metadata_noc_addr, metadata_cb_addr, metadata_size_bytes);  // read_metadata;

    cb_reserve_back(token_counts_cb_id, 1);
    const uint32_t token_counts_cb_addr = get_read_ptr(dense_metadata_cb_id);
    const uint64_t token_counts_noc_addr = get_noc_addr(0, metadata_addrgen);
    noc_async_read(token_counts_noc_addr, token_counts_cb_addr, token_counts_buffer_size_bytes);  // read_token_counts;

    noc_async_read_barrier();

    const uint32_t metadata_addr = get_read_ptr(dense_metadata_cb_id);
    auto* metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_addr);
    auto* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_addr);

    uint32_t metadata_start_idx_offset, token_end;
    int32_t edt[num_local_experts];
    detail::token_work_split<
        global_num_tokens,
        select_experts_k,
        num_local_experts,
        num_token_parallel_cores,
        metadata_entry_size>(
        counts_ptr, metadata_ptr, token_parallel_core_id, metadata_start_idx_offset, token_end, edt);

    // this is cheesy, but pass token starts and ends to the reader through the end of the metadata buffer
    metadata_ptr[global_num_tokens * metadata_entry_size] = metadata_start_idx_offset;
    metadata_ptr[global_num_tokens * metadata_entry_size + 1] = token_end;

    DPRINT << "token_parallel_core_id: " << token_parallel_core_id
           << " metadata_start_idx_offset: " << metadata_start_idx_offset << " token_end: " << token_end
           << " edt[0]: " << edt[0] << " edt[1]: " << edt[1] << "\n";
    for (uint8_t e = 0; e < num_local_experts; ++e) {
        metadata_ptr[global_num_tokens * metadata_entry_size + 2 + e] = static_cast<uint32_t>(edt[e]);
    }

    cb_push_back(dense_metadata_cb_id, 1);
}

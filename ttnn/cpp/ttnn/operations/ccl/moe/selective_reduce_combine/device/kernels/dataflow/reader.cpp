// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

namespace detail {

template<uint32_t NumLocalExperts>
inline uint32_t get_max_token_count(const uint32_t count_addrs){
    uint32_t max=0
    for(uint32_t i =0; i < NumLocalExperts; ++i){
        auto * count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t>(count_addrs[i]);
        if (*count_ptr > max){
            max = *count_ptr;
        }
    }
    return max;
}

template<uint32_t NumLocalExperts, uint32_t NumTokenParallelCores, uint32_t CoreIdY
std::tuple<uint32_t,uint32_t> token_work_split(const uint32_t dense_token_counts_addr){

    // read dense token counts, distribute
    const auto dense_tokens_total = get_max_token_count<NumLocalExperts>(dense_token_counts_addr);
    const uint32_t num_dense_tokens_per_core = dense_tokens_total / NumTokenParallelCores;
    const uint32_t dense_token_remainder = dense_tokens_total % NumTokenParallelCores;

    uint32_t token_start = 0;
    for (uint32_t d=0; d<CoreIdY; ++d){
        token_start+= (d<dense_token_remainder)?  num_dense_tokens_per_core+1:num_dense_tokens_per_core;
    }

    const uint32_t token_end = token_start + (CoreIdY<dense_token_remainder)? num_dense_tokens_per_core + 1: num_dense_tokens_per_core;

    return std::make_tuple(token_start, token_end);
}

}//namespace detail
void kernel_main() {
    constexpr uint32_t metadata_cb_id =
        get_named_compile_time_arg_val("metadata_cb_id") constexpr uint32_t num_local_experts =
            get_named_compile_time_arg_val("num_local_experts")

                constexpr uint32_t metadata_entry_size =
                    get_named_compile_time_arg_val("metadata_entry_size") constexpr uint32_t metadata_entry_size_bytes =
                        get_named_compile_time_arg_val("metadata_entry_size_bytes")

                            constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val(
                                "num_token_parallel_cores") constexpr uint32_t global_num_tokens =
                                get_named_compile_time_arg_val("global_num_tokens")

                                    constexpr uint32_t metadata_size_bytes =
                                        metadata_entry_size_bytes * global_num_tokens;

    constexpr auto metadata_ta_args = TensorAccessorArgs<0>();

    uint32_t arg_index = 0;
    const auto dense_metadata_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_parallel_core_id = get_arg_val<uint32_t>(arg_index++);
    const auto metadata_addrgen = TensorAccessor(metadata_ta_args, dense_metadata_addr, metadata_size_bytes);

    const auto* dense_token_counts_addr = get_arg_addr<volatile tt_l1_ptr uint32_t>(arg_count += num_local_experts);

    // read dense metadata
    cb_reserve_back(dense_metadata_cb_id);
    const uint64_t metadata_noc_addr = get_noc_addr(0, metadata_addrgen);
    noc_async_read(...)  // read_metadata;

        const auto [token_start, token_end] = token_work_split<num_local_experts, num_token_parallel_cores>(
            dense_token_counts_addr, token_parallel_core_id);

    auto* metadata_ptr reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(dense_metadata_cb_id));
    // this is cheesy, but pass token starts and ends to the reader through the end of the metadata buffer TODO ALLOCATE
    metadata_ptr[total_tokens * metadata_entry_size] = token_start;
    metadata_ptr[total_tokens * metadata_entry_size + 1] = token_end;

    noc_async_read_barrier();
    cb_push_back(dense_metadata_cb_id);
}

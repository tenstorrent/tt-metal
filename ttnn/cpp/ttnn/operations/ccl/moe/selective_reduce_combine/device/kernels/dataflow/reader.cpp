// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

namespace detail {




template<uint32_t Limit>
inline bool increment_counter(uint32_t & counter){
    if (counter == Limit-1){
        return true
    }
    else{
        ++counter;
        return false;
    }
}

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


    constexpr uint32_t dense_data_cb_id = get_named_compile_time_arg_val("dense_data_cb_id");
    constexpr uint32_t dense_metadata_cb_id = get_named_compile_time_arg_val("dense_metadata_cb_id")
    
    constexpr uint32_t metadata_entry_size =get_named_compile_time_arg_val("metadata_entry_size")
        
    constexpr uint32_t token_block_height = get_named_compile_time_arg_val
    constexpr uint32_t core_id_y = get_named_compile_time_arg_val // token parallelism
    constexpr uint32_t core_id_x = get_named_compile_time_arg_val // data parallelism

    constexpr uint32_t token_size_bytes = get_named_compile_time_arg_val

    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val
    constexpr uint32_t num_data_parallel_cores = get_named_compile_time_arg_val
    
    constexpr uint32_t source_token_segments = get_named_compile_time_arg_val // 28? 
    constexpr uint32_t source_token_segment_bytes =  token_size_bytes/source_token_segments; // evenly divisible
    constexpr uint32_t source_segments_per_segment = source_token_segments / num_data_parallel_cores;
        
    constexpr uint32_t select_experts_k
    constexpr uint32_t total_tokens
    constexpr auto inactive_expert_val = select_experts_k + 1;
    
    uint32_t arg_index=0;
    const auto num_local_experts = get_arg_val<uint32_t>(arg_index++);
    const auto dense_data_addr = get_arg_val<uint32_t>(arg_index++); 
    const auto dense_metadata_addr = get_arg_val<uint32_t>(arg_index++);
    const auto token_segment_bytes= get_arg_val<uint32_t>(arg_index++); 
    const auto token_segment_offset_bytes = get_arg_val<uint32_t>(arg_index++);
    const auto * dense_token_counts_addr = get_arg_addr<volatile tt_l1_ptr uint32_t>(arg_count+=num_local_experts);


    // read dense metadata
    uint32_t metadata_page = token_start / metadata_entries_per_page;
    cb_reserve_back(dense_metadata_cb_id);
    noc_async_read( ... ) // read_metadata;
    auto * metadata_ptr reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(dense_metadata_cb_id));
    
    // read or prepare data buffer
    
    const auto [token_start, token_end] = 
        token_work_split<num_local_experts, num_token_parallel_cores,core_id_y>(dense_token_counts_addr);
    
    // this is cheesy, but pass token starts and ends to the reader through the end of the metadata buffer TODO ALLOCATE
    metadata_ptr[total_tokens * metadata_entry_size] = token_start;
    metadata_ptr[total_tokens * metadata_entry_size + 1] = token_end;
    
        
    // metadata is uint32_t [token_id, k0, k1, ... k num_local_experts]
    metadata_ptr+=token_start*metadata_entry_size + 1;
    
    noc_async_read_barrier();
    cb_push_back(dense_metadata_cb_id);
    
    // data from untilize is expected [28, 2, batch/32, 32 * 256]

    uint32_t token_offset_bytes = token_start*;
    for (uint32_t t = token_start,metadata_entry_count=token_start%metadata_entries_per_page; t < token_end; ++t){
        
        const auto * k_entries = 
        
        for (uint16_t le = 0; le < num_local_experts; ++le){
            if (k_entries[le] == inactive_expert_val){
                continue;
            }
            cb_reserve_back(reader_cb_id, 1);
            const uint32_t token_base_l1_addr = get_write_ptr(reader_cb_id);
            
            for(uint32_t s = combine_untilize_segment_offset; s < combine_untilize_segment_offset+untilize_segments_per_combine_segment; ++s){
                
                noc_async_read(...);
            }
            
            noc_async_read_barrier();
            cb_push_back(reader_cb_id);
        }
        
        if(token_offset_bytes == untilize_token_segment_bytes*(token_block_height-1)){
            token_offset_bytes=0;
        }
        else{
            token_offset_bytes+=untilize_token_segment_bytes;
        }
        
        metadata_ptr+=metadata_entry_size;
    }  
}
}
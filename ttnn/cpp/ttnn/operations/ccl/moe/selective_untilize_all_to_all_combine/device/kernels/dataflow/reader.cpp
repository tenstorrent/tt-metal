
constexpr chunk_complete_val = std::numeric_limits<uint32_t>::max();

template<bool is_sync_core=true>
struct ChunkTable{
    
private:
    volatile tt_l1_ptr uint32_t * combine_semaphore_table_ptr;
    // ... worker sync semaphore multicast grid stuff

protected: 
    uint32_t num_chunks_complete;
    uint32_t current_chunks_device
    
public:
    const uint64_t & worker_sync_semaphore_addr;
        
    const uint32_t & next() {
        while(true)){

            const uint32_t = current_compute_chunk_counter= *compute_chunk_counter_sem;
            for(uint32_t d=0; d<num_cluster_devices;++d){
                if(combine_semaphore_table_ptr[d]==chunk_complete_val){
                    continue;
                }
                for(uint32_t e=0; e< num_local_experts;++e){
                    // need to confirm how this is laid out
                    if(combine_semaphore_table_ptr[e*num_cluster_devices+d] >= current_compute_chunk_counter){
                        break
                    }
                    else if(e==num_local_experts-1){
                        current_chunks_device = d;
                        // !TODO MULTICAST TO WORKERS
                        return current_chunks_device;
                    }
                }
            }
        }
    
    }
    
    void finish(){
        for (uint32_t e=0; e<num_local_experts; ++e){
            combine_semaphore_table_ptr[e*num_cluster_devices+current_chunks_device]=chunk_complete_val;
        }
        ++num_chunks_complete;
    }
    
    bool done() const {
        return num_chunks_complete==total_chunks;
    }

};

struct ChunkTable: ChunkTable<false>{
    using ChunkTable<false>::current_chunks_device;
    
    const uint32_t & next() {
        
    }
}

inline void gather_chunk_data<uint32_t num_local_experts>(
    const uint32_t sender_device_id, const auto & dense_contribs_accessor, const auto & metadata_accessor){
    
    const sender_token_offset = sender_device_id*batch_per_cluster_device;
    
    
    
    for (uint32_t e0=0; e0<num_local_experts; ++e0)
    for (uint32_t blk=0; blk < num_width_blocks_core; ++blk)    
    for (uint32_t ht = 0; ht < num_height_tiles_per_chunk;++ht){
        
        
        
        cb_reserve_back(reader_cb_id, num_local_experts);
        // Zero ^ N+1 to N 
        const uint32_t l1_base_addr = get_write_ptr(reader_cb_id);
        
        // read the whole tile from this segment of the e0 contribution
        const uint32_t e0_tile_idx = ...
        const uint64_t e0_tile_noc_add = get_noc_addr(e0_tile_idx, accessor);
        noc_async_read(e0_tile_noc_add, l1_base_addr, tile_size_bytes);
        
        uint32_t token_e0_tile_row_idx=0;
        for (uint32_t tok=ht*tile_height; tok < (ht+1)*tile_height;++tok){
            
            if(!is_active(tok,e0, metadata){
                continue;
            }
            
            for (uint32_t e1 = e0+1; le < num_local_experts-1; ++e1){
            
                if(!is_active(tok,e1, metadata)){
                    continue;
                }
                
                uint32_t token_e1_tile_row_idx=get_tile_row(e1, tok, indices);
                   
                const uint32_t reader_l1_addr=get_write_ptr(reader_cb_id);
                
                const uint64_t tile_noc_addr = get_noc_addr(expert_tile_offset+t, accessor);
                const uint32_t e1_l1_addr = reader_cb_base_addr+e1*tile_size_bytes;
                noc_async_read_tile_row(dense_inputs_addr, e1_l1_addr, token_e1_tile_row_idx, token_e0_tile_row_idx);
            }
            
            ++token_tile_row_idx;
        }
        noc_async_read_barrier();
        cb_push_back(reader_cb_id,block_size_tiles);
    }
} 


void kernel_main(){
    
    ChunkTable<is_sync_core> chunk_table(...);
    cb_reserve_back(metadata_cb_id,1);
    auto * metadata_l1_addr reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(metadata_cb_id));
    
    while (!chunk_table.done()){
        
        synchronize_workers<is_sync_core>(...);
        const auto & sender_device_id = chunk_table.next();
        
        read_chunk_tiles(send_device_id);
        
        chunk_table.finish();    
    }
}
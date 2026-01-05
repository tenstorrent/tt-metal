// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "dataflow_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reshuffle.h"

template<uint32_t block_size_tiles, uint32_t num_local_experts, uint32_t reader_cb_id>
inline void accumulate_expert_contributions(const uint32_t bt, const uint32_t e0){

    if constexpr(num_local_experts==1){
        copy_tiles_init(reader_cb_id);
        copy_tile(reader_cb_id, 0, bt+block_size_tiles);
    
    }
    else {
        add_tiles_init(reader_cb_id)
        add_tiles(reader_cb_id, reader_cb_id, e0, e0+1, bt);
        
        if constexpr(num_local_experts > 2){            
            binary_dest_reuse_tiles_init(reader_cb_id);
            for (uint32_t e1 = e0+2; e1 < num_local_experts; ++e1){
                add_tiles_reuse_dest(reader_cb_id, reader_cb_id, e1, e1, bt);
            }
        }
    }
}


void kernel_main(){

    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_height_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cluster_size = get_compile_time_arg_val(2);
    constexpr uint32_t core_idx = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(6);
    constexpr uint32_t num_width_blocks_core = get_compile_time_arg_val(7);
    constexpr uint32_t tile_width_elm = get_compile_time_arg_val(8);
    
    constexpr uint32_t reader_cb_id = get_compile_time_arg_val(10);
    constexpr uint32_t writer_cb_id = get_compile_time_arg_val(12);
    
    // full width of hidden dim in tiles. I think this is what is needed by untilize_block
    constexpr uint32_t num_width_tiles = block_size_tiles*num_width_blocks_core*num_cores;
    
    
    for (uint32_t dc=0; dc < cluster_size; ++dc) // the order in which we receive device chunks is ND
    for (uint32_t e0=0; e0< num_local_experts; ++e0)
    for (uint32_t blk=0; blk < num_width_blocks_core; ++blk){
    
    // TODO tile height?
                
        acquire_dst();
        
        for(uint32_t bt =0;bt<block_size_tiles){
            cb_wait_front(reader_cb_id, num_local_experts);
            
            accumulate_expert_contributions<block_size_tiles, num_local_experts, reader_cb_id>(bt,e0);
            
            cb_pop_front(reader_cb_id, num_local_experts);
        }
           
        tile_regs_commit();
        tile_regs_wait();
           
        // now untilize
        cb_reserve_back(writer_cb_id, block_size_tiles);
                    
        const uint32_t block_c_index = core_idx*num_blocks_per_core + blk*block_size_tiles*tile_width_elm;
        pack_untilize_dest_init<block_size_tiles, full_ct_dim>(writer_cb_id);
        pack_untilize_dest<block_size_tiles, num_width_tiles>(writer_cb_id,1/*block_rt_dim*/,block_c_index);
        tile_regs_release();
        
        cb_push_back(writer_cb_id, block_size_tiles);
           
    }
} 

            
                
            
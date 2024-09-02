// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t is_top_row                     = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma                       = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta                        = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_first_stage         = get_compile_time_arg_val(3);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_const                  = get_compile_time_arg_val(4);
    volatile uint32_t block_h_volatile                = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w_const               = get_compile_time_arg_val(6);
    volatile uint32_t subblock_w_volatile             = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(7);
    const bool is_allgather_worker                    = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block            = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE                      = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t num_blocks_second_stage        = get_compile_time_arg_val(11);

    const uint32_t num_reduce_tiles_per_block_h             = get_arg_val<uint32_t>(0); // This value is the same for all cores, except ones that have padding tiles in it. In that case, skip reduce for padding tiles.
    const uint32_t num_tiles_per_allgather_worker           = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;
    const bool use_two_stage_reduce                         = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader                       = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }

    bool enable_sqrt;
    if (use_two_stage_reduce and not is_second_stage_reader) {
        enable_sqrt = false;
    } else {
        enable_sqrt = true;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_eps = tt::CB::c_in3;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_x = tt::CB::c_intermed0; // x minus mean
    #if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t cb_xmm = cb_in0; // x minus mean
    #else
    constexpr uint32_t cb_xmm = tt::CB::c_intermed1; // x minus mean
    #endif
    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2; // E[x] partials recieved from other cores
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[x^2] partial reduce
    constexpr uint32_t cb_stats = tt::CB::c_in7; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_stats2 = tt::CB::c_intermed4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5; // E[x^2] partials recieved from other cores
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_ex2_global = tt::CB::dataflow6; // E[x^2] global reduce
    constexpr uint32_t cb_x2 = cb_x; // x^2
    constexpr uint32_t cb_reciprocal = tt::CB::c_intermed3; // [E[x^2]-E[x]^2]+eps
    constexpr uint32_t cb_fusion = tt::CB::c_intermed1; // stream gamma/beta
    constexpr uint32_t cb_out = tt::CB::c_out0;

    #ifdef RMSNORM
    constexpr uint32_t cb_var = cb_stats;
    constexpr uint32_t stats_tiles = 1;
    #else
    constexpr uint32_t cb_var = tt::CB::c_intermed2; // Var(x)
    constexpr uint32_t stats_tiles = 2;
    #endif
    constexpr uint32_t cb_ex_sqr = cb_x2;


    binary_op_init_common(cb_stats, cb_stats, cb_ex_sqr);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;



    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr(is_allgather_worker) {
        if (enable_sqrt) {
            cb_reserve_back(cb_stats2, stats_tiles);

            // #ifndef RMSNORM
            // //copy E(x) to cb_stats2

            // tile_regs_acquire();
            // copy_tile_to_dst_init_short();
            // copy_tile(cb_stats, 0, dst0);
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(dst0, cb_stats2);
            // //cb_push_back(cb_stats2, 1);
            // tile_regs_release();

            // // calculate var = E(x^2) - E(x)^2
            // // E(x)^2
            // // unpack_reconfig_data_format(cb_stats, cb_stats);
            // cb_reserve_back(cb_ex_sqr, 1);
            // tile_regs_acquire();
            // mul_tiles_init();
            // mul_tiles(cb_stats, cb_stats, 0, 0, dst0);  // first tile in stats is always E(x)
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(dst0, cb_ex_sqr);
            // cb_push_back(cb_ex_sqr, 1);
            // tile_regs_release();


            // // E(x^2) - E(x)^2
            // unpack_reconfig_data_format_srcb(cb_stats, cb_ex_sqr);
            // pack_reconfig_data_format(cb_var);
            // cb_wait_front(cb_ex_sqr, 1);
            // cb_reserve_back(cb_var, 1);
            // tile_regs_acquire();
            // sub_tiles_init();
            // sub_tiles(cb_stats, cb_ex_sqr, 1, 0, dst0);
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(dst0, cb_var);
            // cb_push_back(cb_var, 1);
            // tile_regs_release();
            // // UNPACK(DPRINT << " cb_var : "<<TSLICE(cb_var, 0, SliceRange::h0_w0_32()) << ENDL());
            // cb_pop_front(cb_ex_sqr, 1);
            // #endif


            // 1/[sqrt(Var + eps)],
            unpack_reconfig_data_format(cb_var, cb_eps);    // cb_var is cb_stats in case of RMS norm
            pack_reconfig_data_format(cb_stats2);
            #ifndef RMSNORM
            cb_wait_front(cb_var, 1);
            #endif
            cb_wait_front(cb_eps, 1);

            // UNPACK(DPRINT << "cb_var : "<<TSLICE(cb_var, 0, SliceRange::h0_w0_32()) << ENDL());
            // UNPACK(DPRINT << "cb_eps : "<<TSLICE(cb_eps, 0, SliceRange::h0_w0_32()) << ENDL());
            //cb_reserve_back(cb_stats2, 1);
            add_tiles_init();
            tile_regs_acquire();
            add_tiles(cb_var, cb_eps, 0, 0, dst0);
            tile_regs_wait();
            // sqrt(Var + eps)
            sqrt_tile_init();
            sqrt_tile(dst0);
            // tile_regs_wait();
            // 1/[sqrt(Var + eps)]
            recip_tile_init();
            recip_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_stats2);

            //cb_pop_front(cb_stats, stats_tiles); // pop the stats tiles we are done and pushed into cb_stats2
            // PACK(DPRINT << " cb_stats E[x] : "<<TSLICE(cb_stats, 0, SliceRange::h0_w0_32()) << ENDL());
            // PACK(DPRINT << "pack cb_stats Reci : "<<TSLICE(cb_ex, 0, SliceRange::h0_w0_32()) << ENDL());
            // UNPACK(DPRINT << "unpack cb_stats Reci : "<<TSLICE(cb_ex, 0, SliceRange::h0_w0_32()) << ENDL());
            cb_pop_front(cb_var, 1);
            cb_pop_front(cb_eps, 1);
            cb_push_back(cb_stats2, stats_tiles);
            tile_regs_release();

            // cb_wait_front(cb_stats2, 1);
            UNPACK(DPRINT << "cb_stats2 : "<<TSLICE(cb_stats2, 0, SliceRange::h0_w0_32()) << ENDL());
        }
    }

    // #ifndef RMSNORM
    // // x - E[x]
    // unpack_reconfig_data_format(cb_in0, cb_ex2_global);
    // pack_reconfig_data_format(cb_xmm);
    // index_h_offset = 0;
    // sub_bcast_cols_init_short();
    // cb_reserve_back(cb_xmm, num_tiles_per_block);
    // for (uint32_t i = 0; i < block_h; i++) {
    //     index_subblock_w_offset = 0;
    //     cb_wait_front(cb_ex2_global, 1);
    //     for (uint32_t j = 0; j < num_subblocks_w; j++) {
    //         tile_regs_acquire();
    //         for (uint32_t w = 0; w < subblock_w; w++) {
    //             index = w + index_subblock_w_offset;
    //             sub_tiles_bcast_cols(cb_in0, cb_ex2_global, index, 0, w);
    //         }
    //         tile_regs_commit();
    //         tile_regs_wait();
    //         for (uint32_t i = 0; i < subblock_w; i++) {
    //             pack_tile(i, cb_xmm);
    //         }
    //         tile_regs_release();
    //         index_subblock_w_offset += subblock_w;
    //     }
    //     cb_pop_front(cb_ex2_global, 1);
    //     cb_pop_front(cb_in0, block_w);
    // }
    // cb_push_back(cb_xmm, num_tiles_per_block);
    // // UNPACK(DPRINT << " cb_xmm X-E[x] : "<<TSLICE(cb_xmm, 0, SliceRange::h0_w0_32()) << ENDL());
    // #endif

    if constexpr(do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }
    // (x - Ex) * 1/[sqrt(Var + eps)]

    unpack_reconfig_data_format(cb_xmm, cb_ex2_global);
    mul_bcast_cols_init_short();
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex2_global, 1);
        //UNPACK(DPRINT << " cb_ex2_global Reci : "<<TSLICE(cb_ex2_global, 0, SliceRange::h0_w0_32()) << ENDL());
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_ex2_global, index, 0, w);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_im);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
        cb_pop_front(cb_ex2_global, 1);
    }
    cb_push_back(cb_im, num_tiles_per_block);

    cb_pop_front(cb_xmm, num_tiles_per_block);
    cb_wait_front(cb_im, num_tiles_per_block);

    if constexpr(do_gamma) {
        unpack_reconfig_data_format(cb_im, cb_gamma);
        if constexpr(do_beta == 0) {
            pack_reconfig_data_format(cb_out);
        }
        mul_bcast_rows_init_short();
        cb_wait_front(cb_gamma, block_w);
        index_h_offset = 0;
        cb_reserve_back(cb_outgamma, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im, cb_gamma, index+index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_outgamma);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_push_back(cb_outgamma, num_tiles_per_block);
        cb_pop_front(cb_im, num_tiles_per_block);
        cb_wait_front(cb_outgamma, num_tiles_per_block);
    }

    if constexpr(do_beta) {
        unpack_reconfig_data_format(cb_fusion, cb_beta);
        pack_reconfig_data_format(cb_out);
        add_bcast_rows_init_short();
        cb_wait_front(cb_beta, block_w);
        index_h_offset = 0;
        cb_reserve_back(cb_out, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion, cb_beta, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_push_back(cb_out, num_tiles_per_block);
        cb_pop_front(cb_fusion, num_tiles_per_block);
        cb_wait_front(cb_out, num_tiles_per_block);
    }
}

}

#include <cstdint>
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    uint32_t num_blocks_m = get_arg_val<uint32_t>(0);
    uint32_t num_blocks_k = get_arg_val<uint32_t>(1);
    uint32_t num_blocks_n = get_arg_val<uint32_t>(2);
    uint32_t sub_block_m = get_arg_val<uint32_t>(3);
    uint32_t sub_block_k = get_arg_val<uint32_t>(4);
    uint32_t sub_block_n = get_arg_val<uint32_t>(5);
    uint32_t dst_size = get_arg_val<uint32_t>(6);
    uint32_t k_block_A = get_arg_val<uint32_t>(7);
    uint32_t k_block_B = get_arg_val<uint32_t>(8);
    uint32_t out_block_tiles = get_arg_val<uint32_t>(9);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    cb_reserve_back(cb_out, out_block_tiles);

    for (uint32_t block_iter_k = 0; block_iter_k < num_blocks_k; block_iter_k++) {
        cb_wait_front(cb_in0, k_block_A);
        cb_wait_front(cb_in1, k_block_B);

        if (block_iter_k == 0) {
            pack_reconfig_l1_acc(false);
        } else if (block_iter_k == 1) {
            pack_reconfig_l1_acc(true);
        }

        for (uint32_t block_iter_m = 0; block_iter_m < num_blocks_m; block_iter_m++) {
            for (uint32_t block_iter_n = 0; block_iter_n < num_blocks_n; block_iter_n++) {
                // Start accumulation here
                tile_regs_acquire();

                for (uint32_t sub_block_iter_k = 0; sub_block_iter_k < sub_block_k; sub_block_iter_k++) {
                    for (uint32_t sub_block_iter_m = 0; sub_block_iter_m < sub_block_m; sub_block_iter_m++) {
                        for (uint32_t sub_block_iter_n = 0; sub_block_iter_n < sub_block_n; sub_block_iter_n++) {
                            uint32_t in0_index =
                                (block_iter_m * sub_block_m + sub_block_iter_m) * sub_block_k + sub_block_iter_k;
                            uint32_t in1_index = sub_block_iter_k * (num_blocks_n * sub_block_n) +
                                                 block_iter_n * sub_block_n + sub_block_iter_n;
                            uint32_t dst_index = sub_block_iter_m * sub_block_n + sub_block_iter_n;

                            matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index);
                        }
                    }
                }

                tile_regs_commit();
                tile_regs_wait();

                pack_tile_block(0, cb_out, dst_size);

                tile_regs_release();
            }
        }

        cb_pop_front(cb_in0, k_block_A);
        cb_pop_front(cb_in1, k_block_B);
    }

    cb_push_back(cb_out, out_block_tiles);

    // for (uint32_t y = top; y < bot; y++) {
    //     for (uint32_t x = left; x < right; x++) {
    //         tile_regs_acquire();
    //         for (uint32_t kt = 0; kt < Kt; kt++) {
    //             // uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t
    //             // idst) {
    //             uint32_t in0_index = (y - top) * Kt + kt;
    //             uint32_t in1_index = kt * (right - left) + (x - left);
    //             matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, 0);
    //         }
    //         tile_regs_commit();
    //         tile_regs_wait();

    //         cb_reserve_back(cb_out, 1);
    //         pack_tile(0, cb_out);
    //         cb_push_back(cb_out, 1);

    //         tile_regs_release();
    //     }
    // }
    // cb_pop_front(cb_in0, block_size_A);
    // cb_pop_front(cb_in1, block_size_B);

    // // TODO: fix block sizes
    // cb_reserve_back(cb_id_in0, block_size_A);
    // uint32_t in0_l1_write_addr = get_write_ptr(cb_id_in0);
    // const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    // for (uint32_t block_iter_m = top; block_iter_m < bot; block_iter_m += sub_block_m) {
    //     for (uint32_t sub_block_iter_m = block_iter_m; sub_block_iter_m < block_iter_m + sub_block_m; block_iter_m++)
    //     {
    //         const uint32_t row = sub_block_iter_m * Kt;
    //         for (int kt = 0; kt < Kt; kt++)
    //         {
    //             noc_async_read_page(row + kt, s0, in0_l1_write_addr);
    //             in0_l1_write_addr += in0_tile_bytes;
    //         }
    //     }
    // }

    // cb_reserve_back(cb_id_in1, block_size_B);
    // uint32_t in1_l1_write_addr = get_write_ptr(cb_id_in1);
    // const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    // for (uint32_t kt = 0; kt < Kt; kt++) {
    //     for(uint32_t block_iter_n = left; block_iter_n < right; block_iter_n += sub_block_n) {
    //         const uint32_t row = kt * Nt;
    //         for (uint32_t subblock_iter_n = block_iter_n; subblock_iter_n < block_iter_n + sub_block_n;
    //         subblock_iter_n++)
    //         {
    //             noc_async_read_page(row + sub_block_iter_n, s1, in1_l1_write_addr);
    //             in1_l1_write_addr += in1_tile_bytes;
    //         }
    //     }
    // }
}

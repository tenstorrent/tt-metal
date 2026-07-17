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

    uint32_t top = get_arg_val<uint32_t>(0);
    uint32_t left = get_arg_val<uint32_t>(1);
    uint32_t bot = get_arg_val<uint32_t>(2);
    uint32_t right = get_arg_val<uint32_t>(3);
    uint32_t block_size_A = get_arg_val<uint32_t>(4);
    uint32_t block_size_B = get_arg_val<uint32_t>(5);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    cb_wait_front(cb_in0, block_size_A);
    cb_wait_front(cb_in1, block_size_B);

    for (uint32_t y = top; y < bot; y++) {
        for (uint32_t x = left; x < right; x++) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t
                // idst) {
                uint32_t in0_index = (y - top) * Kt + kt;
                uint32_t in1_index = kt * (right - left) + (x - left);
                matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, 0);
            }
            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
    cb_pop_front(cb_in0, block_size_A);
    cb_pop_front(cb_in1, block_size_B);

    // for (uint32_t y = top; y < bot; y++) {
    //     for (uint32_t x = left; x < right; x++) {
    //         tile_regs_acquire();

    //         for (uint32_t kt = 0; kt < Kt; kt++) {
    //             cb_wait_front(cb_in0, 1);
    //             cb_wait_front(cb_in1, 1);

    //             matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

    //             cb_pop_front(cb_in0, 1);
    //             cb_pop_front(cb_in1, 1);
    //         }

    //         tile_regs_commit();
    //         tile_regs_wait();

    //         cb_reserve_back(cb_out, 1);
    //         pack_tile(0, cb_out);
    //         cb_push_back(cb_out, 1);

    //         tile_regs_release();
    //     }
    // }
}

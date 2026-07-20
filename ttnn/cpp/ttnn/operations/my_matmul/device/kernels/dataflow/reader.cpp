#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t top = get_arg_val<uint32_t>(5);
    uint32_t left = get_arg_val<uint32_t>(6);
    uint32_t bot = get_arg_val<uint32_t>(7);
    uint32_t right = get_arg_val<uint32_t>(8);
    uint32_t block_size_A = get_arg_val<uint32_t>(9);   // per_core_Mt * Kt
    uint32_t block_size_B = get_arg_val<uint32_t>(10);  // Kt * per_core_Nt

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr);

    cb_reserve_back(cb_id_in0, block_size_A);
    uint32_t in0_l1_write_addr = get_write_ptr(cb_id_in0);
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    for (uint32_t y = top; y < bot; y++) {
        const uint32_t row = y * Kt;
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_page(row + kt, s0, in0_l1_write_addr);
            in0_l1_write_addr += in0_tile_bytes;
        }
    }

    cb_reserve_back(cb_id_in1, block_size_B);
    uint32_t in1_l1_write_addr = get_write_ptr(cb_id_in1);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        const uint32_t row = kt * Nt;
        for (uint32_t x = left; x < right; x++) {
            noc_async_read_page(row + x, s1, in1_l1_write_addr);
            in1_l1_write_addr += in1_tile_bytes;
        }
    }
    noc_async_read_barrier();

    cb_push_back(cb_id_in0, block_size_A);
    cb_push_back(cb_id_in1, block_size_B);
}

// // TODO: fix block sizes
// cb_reserve_back(cb_id_in0, block_size_A);
// uint32_t in0_l1_write_addr = get_write_ptr(cb_id_in0);
// const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
// for (uint32_t block_iter_m = top; block_iter_m < bot; block_iter_m += sub_block_m) {
//     for (uint32_t sub_block_iter_m = block_iter_m; sub_block_iter_m < block_iter_m + sub_block_m; block_iter_m++) {
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

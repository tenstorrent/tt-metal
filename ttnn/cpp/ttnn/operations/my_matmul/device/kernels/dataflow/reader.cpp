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
    uint32_t num_blocks_k = get_arg_val<uint32_t>(9);
    uint32_t sub_block_k = get_arg_val<uint32_t>(10);
    uint32_t k_block_A = get_arg_val<uint32_t>(11);
    uint32_t k_block_B = get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr);

    const uint32_t in0_tile_bytes = get_tile_size(cb_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_in1);

    for (uint32_t block_iter_k = 0; block_iter_k < num_blocks_k; block_iter_k++) {
        cb_reserve_back(cb_in0, k_block_A);
        cb_reserve_back(cb_in1, k_block_B);

        uint32_t k_offset = block_iter_k * sub_block_k;
        uint32_t in0_l1_write_addr = get_write_ptr(cb_in0);
        for (uint32_t y = top; y < bot; y++) {
            for (uint32_t kt = k_offset; kt < k_offset + sub_block_k; kt++) {
                noc_async_read_page(y * Kt + kt, s0, in0_l1_write_addr);
                in0_l1_write_addr += in0_tile_bytes;
            }
        }

        uint32_t in1_l1_write_addr = get_write_ptr(cb_in1);
        for (uint32_t kt = k_offset; kt < k_offset + sub_block_k; kt++) {
            const uint32_t row = kt * Nt;
            for (uint32_t x = left; x < right; x++) {
                noc_async_read_page(row + x, s1, in1_l1_write_addr);
                in1_l1_write_addr += in1_tile_bytes;
            }
        }

        noc_async_read_barrier();

        cb_push_back(cb_in0, k_block_A);
        cb_push_back(cb_in1, k_block_B);
    }
}

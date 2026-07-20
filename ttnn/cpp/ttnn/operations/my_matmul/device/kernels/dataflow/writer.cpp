#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_arg_val<uint32_t>(1);
    uint32_t top = get_arg_val<uint32_t>(2);
    uint32_t left = get_arg_val<uint32_t>(3);
    uint32_t num_blocks_m = get_arg_val<uint32_t>(4);
    uint32_t num_blocks_n = get_arg_val<uint32_t>(5);
    uint32_t sub_block_m = get_arg_val<uint32_t>(6);
    uint32_t sub_block_n = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out = 16;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr);

    for (uint32_t block_iter_m = 0; block_iter_m < num_blocks_m; block_iter_m++) {
        for (uint32_t block_iter_n = 0; block_iter_n < num_blocks_n; block_iter_n++) {
            for (uint32_t sub_block_iter_m = 0; sub_block_iter_m < sub_block_m; sub_block_iter_m++) {
                for (uint32_t sub_block_iter_n = 0; sub_block_iter_n < sub_block_n; sub_block_iter_n++) {
                    cb_wait_front(cb_out, 1);
                    uint32_t l1_read_addr = get_read_ptr(cb_out);
                    uint32_t row = top + block_iter_m * sub_block_m + sub_block_iter_m;
                    uint32_t col = left + block_iter_n * sub_block_n + sub_block_iter_n;
                    uint32_t out_addr = row * Nt + col;
                    noc_async_write_page(out_addr, s, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, 1);
                }
            }
        }
    }

    // for (uint32_t y = top; y < bot; y++) {
    //     for (uint32_t x = left; x < right; x++) {
    //         cb_wait_front(cb_id_out, 1);
    //         uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    //         noc_async_write_page(y * Nt + x, s, l1_read_addr);
    //         noc_async_write_barrier();
    //         cb_pop_front(cb_id_out, 1);
    //     }
    // }
}

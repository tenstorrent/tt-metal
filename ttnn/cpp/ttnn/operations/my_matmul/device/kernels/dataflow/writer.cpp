#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);
    uint32_t top = get_arg_val<uint32_t>(3);
    uint32_t left = get_arg_val<uint32_t>(4);
    uint32_t bot = get_arg_val<uint32_t>(5);
    uint32_t right = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out = 16;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr);

    for (uint32_t y = top; y < bot; y++) {
        for (uint32_t x = left; x < right; x++) {
            cb_wait_front(cb_id_out, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_page(y * Nt + x, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}

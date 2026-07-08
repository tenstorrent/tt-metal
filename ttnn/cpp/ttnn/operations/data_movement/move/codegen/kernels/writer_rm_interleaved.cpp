// Writer for RM reshape: sequential stick writes with new page size.
// Uses aligned_page_size for TensorAccessor addressing, new_stick_size for transfer.
//
// CT args: cb_out, new_stick_size, aligned_page_size, TensorAccessorArgs(out_t)
// RT args: dst_addr, num_reads, num_sticks_per_read, num_sticks_per_cb_push, start_stick
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr              = get_arg_val<uint32_t>(0);
    uint32_t num_reads             = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_read   = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_stick           = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t new_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s = TensorAccessor(dst_args, dst_addr, aligned_page_size);

    uint32_t i_stick = start_stick;
    for (uint32_t iter = 0; iter < num_reads; ++iter) {
        cb_wait_front(cb_out0, num_sticks_per_cb_push);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        for (uint32_t i = 0; i < num_sticks_per_read; ++i) {
            noc_async_write_page(i_stick, s, l1_read_addr, new_stick_size);
            l1_read_addr += new_stick_size;
            i_stick += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out0, num_sticks_per_cb_push);
    }
}

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t c_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_c = 2;

    uint32_t tile_bytes_a = get_tile_size(cb_a);
    uint32_t tile_bytes_b = get_tile_size(cb_b);
    uint32_t tile_bytes_c = get_tile_size(cb_c);
    const auto accessor_a = TensorAccessor(a_args, a_addr, tile_bytes_a);
    const auto accessor_b = TensorAccessor(b_args, b_addr, tile_bytes_b);
    const auto accessor_c = TensorAccessor(c_args, c_addr, tile_bytes_c);
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_reserve_back(cb_a, 1);
        uint32_t l1_write_addr_a = get_write_ptr(cb_a);
        noc_async_read_tile(t, accessor_a, l1_write_addr_a);
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        uint32_t l1_write_addr_b = get_write_ptr(cb_b);
        noc_async_read_tile(t, accessor_b, l1_write_addr_b);
        noc_async_read_barrier();
        cb_push_back(cb_b, 1);
        cb_reserve_back(cb_c, 1);
        uint32_t l1_write_addr_c = get_write_ptr(cb_c);
        noc_async_read_tile(t, accessor_c, l1_write_addr_c);
        noc_async_read_barrier();
        cb_push_back(cb_c, 1);
    }
}

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr auto args = TensorAccessorArgs<0>();

    constexpr uint32_t cb = 16;

    uint32_t tile_bytes = get_tile_size(cb);
    const auto accessor = TensorAccessor(args, addr, tile_bytes);
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(cb, 1);
        uint32_t l1_read_addr = get_read_ptr(cb);
        noc_async_write_tile(t, accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb, 1);
    }
}

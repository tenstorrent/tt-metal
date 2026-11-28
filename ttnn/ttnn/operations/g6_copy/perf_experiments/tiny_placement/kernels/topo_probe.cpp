// Topology probe: each core records its own NoC0/NoC1 node ids and the NoC address the
// TensorAccessor resolves for pages 0..7 (banks 0..7) on NoC0 and NoC1, then writes them
// to output page = its linear index.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto args = TensorAccessorArgs<1>();
    const uint32_t addr = get_common_arg_val<uint32_t>(0);
    const uint32_t idx = get_arg_val<uint32_t>(0);
    const auto acc = TensorAccessor(args, addr);
    cb_reserve_back(cb, 1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    p[0] = my_x[0];
    p[1] = my_y[0];
    p[2] = my_x[1];
    p[3] = my_y[1];
    for (uint32_t b = 0; b < 8; ++b) {
        p[4 + b] = (uint32_t)(acc.get_noc_addr(b, 0, 0) >> 32);
        p[12 + b] = (uint32_t)(acc.get_noc_addr(b, 0, 1) >> 32);
    }
    for (uint32_t i = 20; i < 32; ++i) {
        p[i] = 0;
    }
    noc_async_write(get_write_ptr(cb), acc.get_noc_addr(idx), 128);
    noc_async_write_barrier();
}

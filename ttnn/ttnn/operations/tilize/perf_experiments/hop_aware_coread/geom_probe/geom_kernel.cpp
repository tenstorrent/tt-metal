// Geometry dump (hop_aware_coread): per Tensix core, my_x/my_y on both NoCs and the NoC address
// high word of every DRAM bank on both NoCs, written as one 128-B page per core.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto acc = TensorAccessor(args, out_addr, 128);
    cb_reserve_back(0, 1);
    const uint32_t l1 = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    p[0] = my_x[0];
    p[1] = my_y[0];
    p[2] = my_x[1];
    p[3] = my_y[1];
    for (uint32_t b = 0; b < 12; ++b) {
        p[4 + b] = static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(b, 0, 0) >> 32);
        p[16 + b] = static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(b, 0, 1) >> 32);
    }
    p[28] = page;
    p[29] = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);
    p[30] = NOC_CMD_BUF_READ_REG(1, 0, NOC_NODE_ID);
    noc_async_write(l1, acc.get_noc_addr(page), 128);
    noc_async_write_barrier();
}

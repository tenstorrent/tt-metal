// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Vt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Mt = get_compile_time_arg_val(2);
    constexpr auto out_a = TensorAccessorArgs<3>();
    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile_size = get_tile_size(7);
    const auto out_acc = TensorAccessor(out_a, out_addr, tile_size);
    Noc noc;
    CircularBuffer out(7);
    for (uint32_t i = 0; i < wi_count; i++) {
        const uint32_t wi = wi_start + i;
        const uint32_t bh = wi / Mt;
        const uint32_t mt = wi % Mt;
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t out_base = (b * Mt + mt) * H * Vt + h * Vt;
        out.wait_front(Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_write(src, out_acc, tile_size, {.offset_bytes = vt * tile_size}, {.page_id = out_base + vt});
        }
        noc.async_write_barrier();
        out.pop_front(Vt);
    }
}

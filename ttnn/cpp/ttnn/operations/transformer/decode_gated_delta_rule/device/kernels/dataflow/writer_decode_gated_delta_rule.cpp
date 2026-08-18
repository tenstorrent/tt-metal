// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused T=1 decode gated delta rule. Device 2.0 API.
//
// o is [B,1,H,V] TILE: flat [B*H, V] — head bh's output row lives at row
// (bh % 32) of the shared o pages, so each output tile's row 0 is scattered
// back as two 16-elem face-row chunks (inverse of the reader gather). The new
// state [B,H,K,V] is head-aligned and written as full tiles.
//
// Compile args: {Kt, Vt} + accessor args for (o, new_state).
// Runtime args: {bh, o addr, state addr}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 19, cb_sout = 18;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);

    constexpr auto o_a = TensorAccessorArgs<2>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t bh = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);

    const uint32_t tb_io = get_tile_size(cb_out);
    const uint32_t elem = tb_io / 1024;
    const uint32_t chunk = 16 * elem;
    const auto o_acc = TensorAccessor(o_a, o_addr, tb_io);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb_io);

    constexpr uint32_t kv = Kt * Vt;

    Noc noc;

    const uint32_t r = bh % 32;
    const uint32_t frow = r % 16;
    const uint32_t fhalf = r / 16;
    const uint32_t dst_e0 = (fhalf * 2 + 0) * 256 + frow * 16;
    const uint32_t dst_e1 = (fhalf * 2 + 1) * 256 + frow * 16;

    // o: scatter row 0 of each [1,Vt] tile into row r of the shared o pages.
    {
        CircularBuffer cb(cb_out);
        cb.wait_front(Vt);
        const uint32_t src = cb.get_read_ptr();
        const uint32_t row_group = (bh / 32) * Vt;
        for (uint32_t t = 0; t < Vt; t++) {
            const uint32_t page = row_group + t;
            const uint32_t s = src + t * tb_io;
            noc.async_write(
                CoreLocalMem<uint32_t>(s), o_acc, chunk, {.offset_bytes = 0}, {.page_id = page, .offset_bytes = dst_e0 * elem});
            noc.async_write(
                CoreLocalMem<uint32_t>(s),
                o_acc,
                chunk,
                {.offset_bytes = 256 * elem},
                {.page_id = page, .offset_bytes = dst_e1 * elem});
        }
        noc.async_write_barrier();
        cb.pop_front(Vt);
    }

    // new state: full tiles, head-aligned contiguous pages.
    {
        CircularBuffer cb(cb_sout);
        cb.wait_front(kv);
        const uint32_t base_page = bh * kv;
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < kv; t++) {
            noc.async_write(src, s_acc, tb_io, {.offset_bytes = t * tb_io}, {.page_id = base_page + t});
        }
        noc.async_write_barrier();
        cb.pop_front(kv);
    }
}

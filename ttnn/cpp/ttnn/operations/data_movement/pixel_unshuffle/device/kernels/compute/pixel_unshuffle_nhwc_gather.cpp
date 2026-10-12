// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last: gather helpers on the compute RISCs. No math - the UNPACK, MATH and
// PACK threads each copy helper24/24 of every image row that the two dataflow readers stage, with
// the same plain L1 loads and stores the readers use (pixel_unshuffle_nhwc_common.hpp), so five
// RISCs' worth of L1 bandwidth go into the gather instead of two.
//
// Synchronisation is entirely through the CB page counters the runtime zeroes at launch, read and
// written as plain stream registers (no LLK push/pop: those go through Tensix instructions that
// stall on packer/unpacker state this kernel never sets up):
//   reader r  ->  cb_in[r].received  (the row's input is in the page)  ->  all three threads poll it
//   PACK      ->  cb_done[r].received, MATH -> cb_done[r].acked (each: "my share is gathered")
//   UNPACK    ->  waits for both, then writes cb_in[r].acked, which lets reader r reuse the page.
// All threads walk the two readers' item sequences in lock step ((0,n), (1,n), (0,n+1), ...),
// recomputed here from the same geometry and pixel range the readers use.
// The firmware only initialises UNPACK's read-side and PACK's write-side CB interfaces; MATH reads the
// CB addresses straight from the CB config table of the launch message.

#include <cstdint>
#include "api/compute/common.h"
#include "internal/circular_buffer_interface.h"
#include "stream_io_map.h"
#include "internal/circular_buffer_init.h"
#include "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/pixel_unshuffle_nhwc_common.hpp"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t r = get_compile_time_arg_val(3);
    constexpr uint32_t Ho = get_compile_time_arg_val(4);
    constexpr uint32_t Wo = get_compile_time_arg_val(5);
    constexpr uint32_t datum = get_compile_time_arg_val(6);
    constexpr uint32_t Cp = get_compile_time_arg_val(7);
    constexpr uint32_t out_row_nbytes = get_compile_time_arg_val(8);
    constexpr uint32_t aligned_row_nbytes_in = get_compile_time_arg_val(9);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t cb_done0 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_done1 = get_compile_time_arg_val(14);
    constexpr uint32_t channel_order = get_compile_time_arg_val(15);
    constexpr uint32_t helper24 = get_compile_time_arg_val(16);  // 24ths of each span gathered by the three compute threads together
    constexpr uint32_t math24 = get_compile_time_arg_val(17);    // of which MATH takes math24; UNPACK and PACK split the rest
    constexpr uint32_t up24 = (helper24 - math24) / 2;           // UNPACK's share
    constexpr uint32_t pk24 = helper24 - math24 - up24;          // PACK's share

    const uint32_t pix0 = get_arg_val<uint32_t>(0);
    const uint32_t npix = get_arg_val<uint32_t>(1);
    if (npix == 0) {
        return;
    }

    using G = pixel_unshuffle_nhwc::Geom<W, C, H, r, Ho, Wo, datum, Cp, out_row_nbytes, aligned_row_nbytes_in, channel_order>;
    using pixel_unshuffle_nhwc::Item;

    const uint32_t end = pix0 + npix;
    volatile uint32_t* in_received[2] = {get_cb_tiles_received_ptr(cb_in0), get_cb_tiles_received_ptr(cb_in1)};
    volatile uint32_t* in_acked[2] = {get_cb_tiles_acked_ptr(cb_in0), get_cb_tiles_acked_ptr(cb_in1)};
    volatile uint32_t* done_pack[2] = {get_cb_tiles_received_ptr(cb_done0), get_cb_tiles_received_ptr(cb_done1)};
    volatile uint32_t* done_math[2] = {get_cb_tiles_acked_ptr(cb_done0), get_cb_tiles_acked_ptr(cb_done1)};
    auto wait_count = [](volatile uint32_t* p, uint32_t want) {
        const uint16_t w = (uint16_t)want;
        while ((int16_t)((uint16_t)(*p) - w) < 0) {
        }
    };
    // Each compute thread only has its own side of the CB interfaces initialised: UNPACK the read
    // pointers, PACK the write pointers. Both sit at the CB base (nothing here moves them), and page n
    // of a 2-page CB is at base + (n&1)*page.
    uint32_t out_base = 0;
    uint32_t base[2] = {0, 0};
    UNPACK({
        out_base = get_local_cb_interface(cb_out).fifo_rd_ptr << cb_addr_shift;
        base[0] = get_local_cb_interface(cb_in0).fifo_rd_ptr << cb_addr_shift;
        base[1] = get_local_cb_interface(cb_in1).fifo_rd_ptr << cb_addr_shift;
    });
    PACK({
        out_base = get_local_cb_interface(cb_out).fifo_wr_ptr << cb_addr_shift;
        base[0] = get_local_cb_interface(cb_in0).fifo_wr_ptr << cb_addr_shift;
        base[1] = get_local_cb_interface(cb_in1).fifo_wr_ptr << cb_addr_shift;
    });
    MATH({
        tt_l1_ptr mailboxes_t* const mb = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
        launch_msg_t* lm = &(mb->launch[mb->launch_msg_rd_ptr]);
        const uint32_t kcb = lm->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
        uint32_t tt_l1_ptr* cb_l1_base = (uint32_t tt_l1_ptr*)(kcb + lm->kernel_config.local_cb_offset);
        // CB config entry i = {address (bytes), size, num_pages, page_size}; MATH has no cb_interface storage,
        // so the addresses are read straight from the table.
        out_base = cb_l1_base[cb_out * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
        base[0] = cb_l1_base[cb_in0 * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
        base[1] = cb_l1_base[cb_in1 * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
    });

    uint32_t cursor[2] = {pix0, pix0};
    uint32_t n[2] = {0, 0};
    Item it[2];
    bool alive[2];
    alive[0] = G::advance(0, pix0, end, cursor[0], it[0]);
    alive[1] = G::advance(1, pix0, end, cursor[1], it[1]);

    while (alive[0] || alive[1]) {
        for (uint32_t s = 0; s < 2; s++) {
            if (!alive[s]) {
                continue;
            }
            const uint32_t page = base[s] + (n[s] & 1) * G::slot_nbytes;
            uint32_t a = 0;
            uint32_t b = 0;
            UNPACK({
                wait_count(in_received[s], n[s] + 1);
                G::part(it[s], 24 - helper24, 24 - helper24 + up24, a, b);
                G::gather(it[s], a, b, page, out_base);
                wait_count(done_math[s], n[s] + 1);
                wait_count(done_pack[s], n[s] + 1);
                *in_acked[s] = (uint16_t)(n[s] + 1);  // page free for the reader
            });
            MATH({
                wait_count(in_received[s], n[s] + 1);
                G::part(it[s], 24 - helper24 + up24, 24 - pk24, a, b);
                G::gather(it[s], a, b, page, out_base);
                *done_math[s] = (uint16_t)(n[s] + 1);
            });
            PACK({
                wait_count(in_received[s], n[s] + 1);
                G::part(it[s], 24 - pk24, 24, a, b);
                G::gather(it[s], a, b, page, out_base);
                *done_pack[s] = (uint16_t)(n[s] + 1);
            });
            n[s]++;
            alive[s] = G::advance(s, pix0, end, cursor[s], it[s]);
        }
    }
}

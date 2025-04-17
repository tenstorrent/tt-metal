// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <stdint.h>
#include <array>

// #include "debug/dprint.h"  // required in all kernels using DPRINT

constexpr uint32_t round_up_to_multiple_of_64(uint32_t value) { return (value + 63) & ~63; }

void print_uint64(uint64_t number) {
    DPRINT << "uint64: 0x" << HEX() << (number >> 32) << (number & 0xFFFFFFFF) << DEC() << ENDL();
}

void kernel_main() {
    // Deinterleaves input image in src cb to dest cb.
    //
    // The input data is expected to be interleaved in the following way:
    //     A B A B A B
    //     C D C D C D
    //     A B A B A B
    //     C D C D C D
    // The output data is expected to be deinterleaved in the following way:
    //     A A A
    //     A A A
    //     B B B
    //     B B B
    //     C C C
    //     C C C
    //     D D D
    //     D D D
    //
    // Image width, height and number of channels are given as compile time arguments.
    // Kernel processes AB or CD lines depending on

    constexpr uint64_t src_base_address = get_compile_time_arg_val(0);
    constexpr uint32_t src_width = get_compile_time_arg_val(1);
    constexpr uint32_t src_height = get_compile_time_arg_val(2);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t stride_h = get_compile_time_arg_val(4);
    constexpr uint32_t stride_w = get_compile_time_arg_val(5);

    std::array<uint32_t, stride_h / 2 * stride_w> dst_addresses;
    for (uint32_t h = 0; h < stride_h / 2; h++) {  // stride_h / 2 for two DM cores
        for (uint32_t w = 0; w < stride_w; w++) {
            const uint32_t out_idx = h * stride_w + w;
            dst_addresses[out_idx] = get_common_arg_val<uint32_t>(out_idx);

            // DPRINT << "h= " << h << "; w= " << w << "; dst_address[" << out_idx << "] = 0x" << HEX()
            //        << dst_addresses[out_idx] << DEC() << ENDL();
        }
    }
    constexpr uint32_t src_line_size_bytes = src_width * stick_size_bytes;
    // constexpr uint32_t dst_line_size_bytes = width / stride_w * stick_size_bytes;

    // base address would point to first and second row for two DM cores respectively
    const uint64_t src_base_noc_address = get_noc_addr(src_base_address);

    // Copy data from src to dst
    // h/2 as we divide work between two DM cores
    uint64_t src_address = src_base_noc_address;

    noc_async_read_one_packet_set_state(src_base_noc_address, stick_size_bytes);

    for (uint32_t h = 0; h < src_height / 2; h++) {
        // dst_h_selector is the index of the destination address for the current row
        // that is combined with column to get the final output buffer
        uint32_t dst_h_selector = h & (stride_h / 2 - 1);
        // uint32_t dst_h_selector = (h / 2) % stride_h;

        constexpr uint32_t dst_width = src_width / stride_w;
        uint32_t dst_offset = (h / (stride_h / 2)) * dst_width * stick_size_bytes;

        for (uint32_t w = 0; w < src_width; w += stride_w) {
            for (uint32_t datum = 0; datum < stride_w; datum++) {
                const uint32_t dst_idx = dst_h_selector * stride_w + datum;
                // DPRINT << "dst_idx = " << dst_idx << "; datum = " << datum << "; dst_h_selector = " << dst_h_selector
                //        << "; h = " << h << ENDL();

                constexpr uint32_t dst_offset_limit = src_height / stride_h * src_width / stride_w * stick_size_bytes;
                // if (dst_offset > dst_offset_limit)
                // {
                //     DPRINT << "Abort: dst_offset > dst_offset_limit" << ENDL();
                // }
                // else {
                //     DPRINT << "dst_offset = " << dst_offset << "; dst_idx = " << dst_idx << "; h = " << h <<  "; w =
                //     " << w << "; datum = " << datum <<  ENDL();
                // }
                const uint32_t dst_address =
                    dst_addresses[dst_idx] + dst_offset;  // to try increasing dst_addresses[dst_idx] directly
                noc_async_read_one_packet_with_state<true>(src_address, dst_address);
                src_address += stick_size_bytes;
            }
            dst_offset += stick_size_bytes;
            // noc_async_read_one_packet_set_state(src_noc_address, stick_size_bytes);
            // noc_async_read_one_packet_with_state<true>(src_noc_address, );
        }
        // skip one row, that is processed by the other DM core
        src_address += src_line_size_bytes;
        // print_uint64(src_address - src_base_noc_address);
        noc_async_read_barrier();  // to optimize position
    }
    // DPRINT << "stick_size_bytes=" << stick_size_bytes << ENDL();

    // DPRINT << "Deinterleave done" << ENDL();
}

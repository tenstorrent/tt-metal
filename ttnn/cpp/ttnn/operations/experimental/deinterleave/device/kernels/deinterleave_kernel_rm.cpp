// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"  // required in all kernels using DPRINT

#include <cstdint>
#include "risc_common.h"

template <typename T>
void helper_print_cb(const uint32_t cb_id, const uint32_t height, const uint32_t width, const uint32_t stick_size) {
    DPRINT << "dst " << "height=" << height << ";width=" << width << ";stick_size=" << stick_size << ENDL();
    T* cb_ptr = reinterpret_cast<T*>(get_read_ptr(cb_id));
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width; w++) {
            DPRINT << " <";
            for (uint32_t c = 0; c < stick_size; c++) {
                T val = *(cb_ptr + h * width * stick_size + w * stick_size + c);
                // if (val == 0x42c8) val = 0x0100;
                // if (val == 0x4348) val = 0x0200;
                // if (val == 0x4396) val = 0x0300;
                // if (val == 0x43c8) val = 0x0400;
                DPRINT << HEX() << val << DEC() << ",";
            }
            DPRINT << "> ";
        }
        DPRINT << ENDL();
    }
}

template <typename T>
void helper_clear_cb(
    const uint32_t cb_id, const uint32_t height, const uint32_t width, const uint32_t stick_size, const T value) {
    T* dst = reinterpret_cast<T*>(get_write_ptr(cb_id));
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width; w++) {
            for (uint32_t c = 0; c < stick_size; c++) {
                *dst = value;
                dst++;
            }
        }
    }
}

// Target 8KB of data before a single barrier for 8x8 grid of readers
template <uint32_t payload_size, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    // return ((512 / num_readers) * (1024 + 128)) / payload_size;
    // return (32 * 1024 / num_readers) / payload_size;  // should be magic number and compile time argument

    // return 256;
    // marko's magic numbers

    if constexpr (payload_size == 64) {
        return 48;
    } else if constexpr (payload_size == 96) {
        return 32;
    } else if constexpr (payload_size == 128) {
        return 16;
    }
    return 4;
}

void kernel_main() {
    // if (noc_index == 0)
    // {
    //     DPRINT << "EARLY EXIT!" << ENDL();
    //     return;
    // }
    // DPRINT << "NOC" << (uint32_t)noc_index << " " << (uint32_t)my_y[noc_index] << "-" << (uint32_t)my_x[noc_index] <<
    // ENDL();

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t width = get_compile_time_arg_val(2);
    constexpr uint32_t height = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);
    constexpr uint32_t barrier_threshold = get_compile_time_arg_val(7) != 0
                                               ? get_compile_time_arg_val(7)
                                               : get_barrier_read_threshold<stick_size_bytes, 2>();

    static_assert(stick_size_bytes <= NOC_MAX_BURST_SIZE, "stick size too big, cannot use one_packet API for reads");
    const uint32_t start_x = get_arg_val<uint32_t>(0) + VIRTUAL_TENSIX_START_X;  //
    const uint32_t end_x = get_arg_val<uint32_t>(1) + VIRTUAL_TENSIX_START_X;    //
    const uint32_t start_y = get_arg_val<uint32_t>(2) + VIRTUAL_TENSIX_START_Y;  //
    const uint32_t end_y = get_arg_val<uint32_t>(3) + VIRTUAL_TENSIX_START_Y;    //
    const uint32_t src_width_stride = get_arg_val<uint32_t>(4);
    const uint32_t src_height_offset_to_next = get_arg_val<uint32_t>(5);
    const uint32_t src_offset = get_arg_val<uint32_t>(6);
    const uint32_t dst_size_bytes = get_arg_val<uint32_t>(7);
    const uint32_t dst_offset = get_arg_val<uint32_t>(8);
    const uint32_t offset_x = get_arg_val<uint32_t>(9) + VIRTUAL_TENSIX_START_X;
    const uint32_t offset_y = get_arg_val<uint32_t>(10) + VIRTUAL_TENSIX_START_Y;
    const uint32_t num_src_cores = get_arg_val<uint32_t>(11);
    const uint32_t dst_rollover_offset = get_arg_val<uint32_t>(12);

    uint32_t stick_size = stick_size_bytes / 2;

    // constexpr uint32_t barrier_threshold = get_barrier_read_threshold<stick_size_bytes, 2>();
    uint32_t barrier_count = 0;

    // handy for debug
    // helper_print_cb<uint16_t>(src_cb_id, height, width, stick_size);
    // helper_clear_cb<uint16_t>(dst_cb_id, height, width, stick_size, 0);
    // helper_print_cb<uint16_t>(dst_cb_id, height, width, stick_size);

    // if (my_y[noc_index] == 18 && my_x[noc_index] == 21) {
    //     DPRINT << "EARLY EXIT!" << ENDL();
    //     return;
    // }

    // Go through nodes (start_x, start_y) to (end_x, end_y)
    // Copy your stick (dst_batch) to the dst buffer
    // both DM0/DM1 read from all nodes, assuming that's creating uniform load to the NOC
    // they split reading even/odd lines
    DPRINT << "NOC" << (uint32_t)noc_index << "; T=" << barrier_threshold << "; ssb=" << stick_size_bytes << ENDL();
    auto dst_address = get_write_ptr(dst_cb_id) + dst_offset;

    uint32_t src_noc_x = offset_x;
    uint32_t src_noc_y = offset_y;

    // DPRINT << "num_src_cores = " << num_src_cores
    //     << "; s=(" << start_y << "-" << start_x << ")"
    //     << "; e=(" << end_y << "-" <<end_x << ")"
    //     << "; o=(" << offset_y << "-" << offset_x << ")" << ENDL();

    for (uint32_t src_core = 0; src_core < num_src_cores; src_core += 2) {
        // DPRINT << "** src " << src_noc_y << "-" << src_noc_x << "; dst=" << dst_address - get_write_ptr(dst_cb_id) <<
        // ENDL(); src_noc_address points to the start of the first line (AB) or the second line (CD)
        auto src_noc_address = get_noc_addr(src_noc_x, src_noc_y, get_read_ptr(src_cb_id)) + src_offset;
        noc_async_read_one_packet_set_state(src_noc_address, stick_size_bytes);
        // Copy half of data data from src to dst
        for (uint32_t h = 0; h < height; h += stride_h) {
            for (uint32_t w = 0; w < width; w += stride_w) {
                noc_async_read_one_packet_with_state<true>(src_noc_address, dst_address);
                src_noc_address += src_width_stride;
                dst_address += stick_size_bytes;
                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
            }
            // skip lines to the next src line
            src_noc_address += src_height_offset_to_next;
        }
        dst_address += dst_size_bytes;  // / 2;  // / 2; // dst_stride - one src image in dst size

        // iterate over src cores, with wrapping
        // to figure out better place for this!
        src_noc_x += 2;
        if (src_noc_x >= end_x) {
            src_noc_x = start_x + src_noc_x - end_x;
            src_noc_y++;
            if (src_noc_y >= end_y) {  // rollover
                src_noc_y = start_y;
                // auto old = dst_address;
                dst_address = get_write_ptr(dst_cb_id) + dst_rollover_offset;
                // DPRINT << "reset dst_address from " << old - get_write_ptr(dst_cb_id) << " to " << dst_address -
                // get_write_ptr(dst_cb_id) << ENDL();
            }
        }

        // // // to do - handle this in a better way!
        // auto upper_limit = get_write_ptr(dst_cb_id) + dst_size_bytes * 4;
        // if (dst_address > upper_limit) {

        //     DPRINT << dst_address << " > " << upper_limit << ENDL();
        //     dst_address = get_write_ptr(dst_cb_id) + (dst_address - upper_limit);
        //     DPRINT << "reset dst_address to " << dst_address << ENDL();
        // }
    }

    noc_async_read_barrier();
    // handy for debug
    // if (noc_index == 1)
    // {
    //     DPRINT << "========================" << ENDL();
    //     helper_print_cb<uint16_t>(dst_cb_id, height, width, stick_size);
    // }

    // DPRINT << "Deinterleave done" << ENDL();
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hw/inc/dataflow_api.h"
#include "tt_metal/hw/inc/utils/utils.h"
#include "firmware_common.h"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_unit_size_in_bytes = get_compile_time_arg_val(2);   // input stick size
    constexpr uint32_t output_unit_size_in_bytes = get_compile_time_arg_val(3);  // output stick size
    constexpr uint32_t num_input_channels = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t num_output_channels = get_compile_time_arg_val(6);
    constexpr uint32_t num_sticks_for_all_riscvs = get_compile_time_arg_val(7);

    constexpr uint32_t num_sticks_for_this_riscv = get_compile_time_arg_val(8);
    constexpr uint32_t current_riscv_stick_starting_index = get_compile_time_arg_val(9);

    uint32_t src_cb_ptr = get_read_ptr(src_cb_id);
    uint32_t dst_cb_ptr = get_write_ptr(dst_cb_id);

    // temp, fixme
    constexpr uint32_t output_width = input_width * num_input_channels / (2 * num_output_channels);
    constexpr uint32_t num_widths_done_by_other_riscvs = current_riscv_stick_starting_index / input_width;

    src_cb_ptr += current_riscv_stick_starting_index * input_unit_size_in_bytes;
    dst_cb_ptr +=
        num_widths_done_by_other_riscvs * output_width * output_unit_size_in_bytes +
        current_riscv_stick_starting_index * 2 *
            output_unit_size_in_bytes;  // move to adequate row in dst in addition to moving to the next sticks

    constexpr uint32_t input_read_bytes = input_unit_size_in_bytes / 2;
    static_assert(input_read_bytes <= NOC_MAX_BURST_SIZE);
    noc_async_read_one_packet_set_state(get_noc_addr(src_cb_ptr), input_read_bytes);

    uint32_t num_input_sticks_read = current_riscv_stick_starting_index - num_widths_done_by_other_riscvs * input_width;
    for (uint32_t i = 0; i < num_sticks_for_this_riscv; i++) {
        uint32_t written_in_dst_stick = 0;
        uint32_t stick_index = 0;

        // copy half of input sticks to dst ptr
        noc_async_read_one_packet_with_state<true>(src_cb_ptr, dst_cb_ptr);
        src_cb_ptr += input_unit_size_in_bytes / 2;

        uint32_t dst_cb_ptr_row_below = dst_cb_ptr + output_width * output_unit_size_in_bytes;
        noc_async_read_one_packet_with_state<true>(src_cb_ptr, dst_cb_ptr_row_below);
        src_cb_ptr += input_unit_size_in_bytes / 2;

        dst_cb_ptr += 2 * output_unit_size_in_bytes;  // we wrote 2 sticks in stick index, move it by 2
        num_input_sticks_read++;
        if (__builtin_expect(num_input_sticks_read == input_width, 0)) {
            num_input_sticks_read = 0;
            // skip row we have just written to
            dst_cb_ptr += output_width * output_unit_size_in_bytes;
        }
    }

    // dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
    // for (uint32_t i = 0; i < 18; i++) {
    //         DPRINT << "ROW:" << ENDL();
    //     for (uint32_t j = 0; j < 130; j++) {
    //         for (int k = 0; k < 1; k++) {
    //             DPRINT << dst_cb_ptr[i * input_unit_size_in_bytes * 130 + j * input_unit_size_in_bytes + k] << ",";
    //         }
    //     }
    //         DPRINT << "ROW END" << ENDL();
    // }
    //     dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
    // for (uint32_t i = 0; i < 18; i++) {
    //         DPRINT << "ROW:" << ENDL();
    //     for (uint32_t j = 0; j < 130; j++) {
    //             DPRINT << dst_cb_ptr[i * stick_num_elements * 130 + j * stick_num_elements + 0] << ",
    //             addr: " <<  (uint32_t) &dst_cb_ptr[i * stick_num_elements * 130 + j *
    //             stick_num_elements + 0] << ",";
    //             //DPRINT << dst_cb_ptr[i * stick_num_elements * 130 + j * stick_num_elements + 1] <<
    //             ",";
    //     }
    //         DPRINT << "ROW END" << ENDL();
    // }
}

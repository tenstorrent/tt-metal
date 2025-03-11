// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt_metal/hw/inc/utils/utils.h"

#include "debug/dprint.h"

#if defined(TRISC_MATH) or defined(TRISC_UNPACK) or defined(TRISC_PACK)
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

#if defined(TRISC_UNPACK) or defined(TRISC_PACK) or defined(TRISC_MATH)
namespace NAMESPACE {
#endif

inline __attribute__((always_inline)) constexpr uint32_t log2_constexpr(uint32_t n) {
    uint32_t p = 0;
    while (n > 1) {
        n >>= 1;
        ++p;
    }
    return p;
}

#if defined(TRISC_MATH) or defined(TRISC_UNPACK) or defined(TRISC_PACK)
void MAIN{
#else
void kernel_main() {
#endif

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t input_unit_size_in_bytes = get_compile_time_arg_val(2);   // input stick size
constexpr uint32_t output_unit_size_in_bytes = get_compile_time_arg_val(3);  // output stick size
constexpr uint32_t num_input_channels = get_compile_time_arg_val(4);
constexpr uint32_t input_width = get_compile_time_arg_val(5);
constexpr uint32_t num_output_channels = get_compile_time_arg_val(6);
constexpr uint32_t num_sticks_for_all_riscvs = get_compile_time_arg_val(7);

#if !defined(TRISC_MATH) && !defined(TRISC_PACK)
constexpr uint32_t num_sticks_for_this_riscv = get_compile_time_arg_val(8);
constexpr uint32_t current_riscv_stick_starting_index = get_compile_time_arg_val(9);
#endif

#if defined(TRISC_MATH)
constexpr uint32_t num_sticks_for_this_riscv = get_compile_time_arg_val(10);
constexpr uint32_t current_riscv_stick_starting_index = get_compile_time_arg_val(11);
#endif

#if defined(TRISC_PACK)
constexpr uint32_t num_sticks_for_this_riscv = get_compile_time_arg_val(12);
constexpr uint32_t current_riscv_stick_starting_index = get_compile_time_arg_val(13);
#endif

volatile tt_l1_ptr uint16_t* src_cb_ptr = nullptr;
volatile tt_l1_ptr uint16_t* dst_cb_ptr = nullptr;

#if !defined(TRISC_MATH) and !defined(TRISC_PACK) and !defined(TRISC_UNPACK)
src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(src_cb_id));
dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(dst_cb_id));
#endif

// Math/pack/unpack threads do not have access to get_write_ptr, so we need to pass it:
// DM to unpack, unpack passes it to math, and math passes it to pack
// Do it with a handshake with math thread:
// math sends a signal to unpack
// unpack reads the signal and sends the pointer to math
// math gets the signal
// Unfortunately, we can't use random L1 address to do so, because of kernel prefetching
// so we need to use a safe address
// This is the safe address:
// auto buf = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
// uint32_t* data = buf->data;
// This makes the use of watcher debug ring buffer feature obsolete.
#if defined(TRISC_MATH) or defined(TRISC_UNPACK) or defined(TRISC_PACK) or defined(FIRST_DM_KERNEL)
auto buf = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
volatile uint32_t* data = (volatile uint32_t*)buf->data;

constexpr uint32_t signal_dm_unp = 0x00deba01;
constexpr uint32_t signal_unp_math_1 = 0x00baba01;
constexpr uint32_t signal_unp_math_2 = 0x00baba02;
constexpr uint32_t signal_math_pack_1 = 0x00deda01;
constexpr uint32_t signal_math_pack_2 = 0x00deda02;
#endif

#if defined(FIRST_DM_KERNEL)
while (*data != signal_dm_unp);  // wait for signal_dm_unp from unpack
*data = (uint32_t)src_cb_ptr;    // send the pointer
while (*data != signal_dm_unp);  // wait for receive
*data = (uint32_t)dst_cb_ptr;    // send the pointer
#endif

#if defined(TRISC_MATH)
while (*data != signal_unp_math_2);                                  // wait for unpack to send the pointer
*data = signal_unp_math_1;                                           // send signal_unp_math_1 to unpack
while (*data == signal_unp_math_1);                                  // wait for unpack to send the pointer
src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(*data);  // fetch the pointer
*data = signal_unp_math_1;                                           // send signal_unp_math_1 to unpack
while (*data == signal_unp_math_1);                                  // wait for unpack to send the pointer
dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(*data);  // fetch the pointer
// complete

// signal pack we ready to send data
*data = signal_math_pack_1;
while (*data != signal_math_pack_2);  // wait for signal_math_pack_2 from pack
*data = (uint32_t)src_cb_ptr;         // send the pointer
while (*data != signal_math_pack_2);  // wait for receive
*data = (uint32_t)dst_cb_ptr;         // send the pointer
// complete
#endif

#if defined(TRISC_UNPACK)
*data = signal_dm_unp;           // send signal_dm_unp that we are ready to accept data
while (*data == signal_dm_unp);  // wait for dm to send the pointer
src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(*data);  // fetch the pointer
*data = signal_dm_unp;           // send signal_dm_unp that we are ready to accept data
while (*data == signal_dm_unp);  // wait for dm to send the pointer
dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(*data);  // fetch the pointer
// complete

*data = signal_unp_math_2;           // send signal_unp_math_2 to math
while (*data != signal_unp_math_1);  // wait for signal_unp_math_1 from math
*data = (uint32_t)src_cb_ptr;        // write the pointer
while (*data != signal_unp_math_1);  // wait for signal_unp_math_1 from math
*data = (uint32_t)dst_cb_ptr;        // write the pointer
// complete
#endif

#if defined(TRISC_PACK)
while (*data != signal_math_pack_1);  // wait for signal_math_pack_1 from math
*data = signal_math_pack_2;           // send signal_math_pack_2 to math that we are ready to accept data
while (*data == signal_math_pack_2);  // wait for math to send the pointer
src_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(*data);  // fetch the pointer
*data = signal_math_pack_2;                                          // signal math we ready for next pointer
while (*data == signal_math_pack_2);                                 // wait for math to send the pointer
dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(*data);  // fetch the pointer
// complete
#endif

// temp, fixme
constexpr uint32_t input_stick_num_elements = input_unit_size_in_bytes / 2;    // assuming hardcoded bfloat16
constexpr uint32_t output_stick_num_elements = output_unit_size_in_bytes / 2;  // assuming hardcoded bfloat16
constexpr uint32_t output_width = input_width * num_input_channels / (2 * num_output_channels);
constexpr uint32_t num_elements_to_write_in_dst_stick = num_output_channels;
constexpr uint32_t half_of_input_channels = num_input_channels / 2;
constexpr uint32_t num_widths_done_by_other_riscvs = current_riscv_stick_starting_index / input_width;
constexpr bool num_elements_to_write_in_dst_stick_is_power_of_2 = is_power_of_2(num_elements_to_write_in_dst_stick);

src_cb_ptr += current_riscv_stick_starting_index * input_stick_num_elements;
dst_cb_ptr += num_widths_done_by_other_riscvs * output_width * output_stick_num_elements +
              current_riscv_stick_starting_index * 2 *
                  output_stick_num_elements;  // move to adequate row in dst in addition to moving to the next sticks

uint32_t num_input_sticks_read = current_riscv_stick_starting_index - num_widths_done_by_other_riscvs * input_width;
for (uint32_t i = 0; i < num_sticks_for_this_riscv; i++) {
    uint32_t written_in_dst_stick = 0;
    uint32_t stick_index = 0;

    // Copy the second half of the sticks to the destination in the row below
    volatile tt_l1_ptr uint16_t* dst_cb_ptr_row_below = dst_cb_ptr + output_width * output_stick_num_elements;

#pragma GCC unroll 16
        for (uint32_t j = 0; j < half_of_input_channels; j++) {
            uint32_t dst_index = stick_index * output_stick_num_elements + written_in_dst_stick;
            dst_cb_ptr[dst_index] = src_cb_ptr[j];
            dst_cb_ptr_row_below[dst_index] = src_cb_ptr[j + half_of_input_channels];
            written_in_dst_stick++;

            if constexpr (num_elements_to_write_in_dst_stick_is_power_of_2) {
                uint16_t temp = written_in_dst_stick;
                stick_index += (temp >> log2_constexpr(num_elements_to_write_in_dst_stick));
                written_in_dst_stick = temp & (num_elements_to_write_in_dst_stick - 1);
            } else {
                stick_index += written_in_dst_stick / num_elements_to_write_in_dst_stick;
                written_in_dst_stick %= num_elements_to_write_in_dst_stick;
            }
        }

        src_cb_ptr += input_stick_num_elements;
        dst_cb_ptr += 2 * output_stick_num_elements;  // we wrote 2 sticks in stick index, move it by 2
        num_input_sticks_read++;

        if (__builtin_expect(num_input_sticks_read == input_width, 0)) {
            num_input_sticks_read = 0;
            // skip row we have just written to
            dst_cb_ptr += output_width * output_stick_num_elements;
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

#if defined(TRISC_UNPACK) or defined(TRISC_PACK) or defined(TRISC_MATH)
}
;  // namespace NAMESPACE
#endif

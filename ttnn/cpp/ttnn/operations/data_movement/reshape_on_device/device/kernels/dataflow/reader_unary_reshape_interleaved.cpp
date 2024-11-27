// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

using uint32_t = std::uint32_t;

// tile index to address
inline uint32_t TADDR(uint32_t ti) { return ti << 11; }

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t input_Wt = get_arg_val<uint32_t>(1);
    uint32_t output_N = get_arg_val<uint32_t>(2);
    uint32_t output_C = get_arg_val<uint32_t>(3);
    uint32_t output_Ht = get_arg_val<uint32_t>(4);
    uint32_t output_Wt = get_arg_val<uint32_t>(5);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t ALIGNMENT = get_compile_time_arg_val(1);

    uint32_t num_sticks_per_input_tile_row = input_Wt << 5;  // Tile height is 32
    uint32_t num_sticks_per_output_tile_row = output_Wt << 5;

    constexpr uint32_t SUBTILE_LINE_BYTES = (16 << 1);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;

    constexpr bool MISALIGNED = ALIGNMENT > SUBTILE_LINE_BYTES;
    uint32_t intermed_l1_scratch = MISALIGNED ? get_write_ptr(1) : 0;
    volatile tt_l1_ptr uint8_t* intermed_l1_scratch_ptr = (volatile uint8_t*)intermed_l1_scratch;

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = tile_bytes, .data_format = data_format};

    // Sticks are a row of elements in a single tile (32 elements)
    // Stick id increments row-wise
    // We loop through output tiles row wise, and then read one tile row at a tile to form the output tile
    uint32_t base_tile_row_stick_id = 0;                // ID of the first stick in the target output tile row
    for (uint32_t n = 0; n < output_N; n++) {           // Iterate over output N
        for (uint32_t c = 0; c < output_C; c++) {       // Iterate over output C TODO: Combine into single loop of batch
            for (uint32_t h = 0; h < output_Ht; h++) {  // Iterate over output Ht
                uint32_t base_tile_stick_id = base_tile_row_stick_id;
                for (uint32_t w = 0; w < output_Wt; w++) {
                    uint32_t output_stick_id = base_tile_stick_id;  // Offset tile id of the current sub tile row
                    cb_reserve_back(cb_id_in0, onetile);
                    for (uint32_t tile_h = 0; tile_h < 32; tile_h++) {
                        uint32_t input_tile_row_to_read = output_stick_id / num_sticks_per_input_tile_row;
                        uint32_t input_tile_col_to_read = output_stick_id % input_Wt;
                        uint32_t input_tile_to_read = input_tile_row_to_read * input_Wt + input_tile_col_to_read;
                        uint32_t input_tile_sub_row_to_read =
                            output_stick_id % num_sticks_per_input_tile_row / input_Wt;

                        uint64_t banked_addr = get_noc_addr(input_tile_to_read, s0);
                        banked_addr +=
                            (((input_tile_sub_row_to_read >> 4) << 1)
                             << 9);  // if intra-tile source h is > 16, add 2*512 to subtile offset
                        banked_addr += ((input_tile_sub_row_to_read & 15) << 5);  // 16 * 2 bytes per face row

                        uint32_t dest_tr0_l1 = get_write_ptr(cb_id_in0);
                        dest_tr0_l1 +=
                            (((tile_h >> 4) << 1) << 9);  // if intra-tile source h is > 16, add 2*512 to subtile offset
                        dest_tr0_l1 += ((tile_h & 15) << 5);  // 16 * 2 bytes per face row

                        for (uint8_t i = 0; i < 2; ++i) {
                            if (MISALIGNED) {
                                // if banked addr and dest addr don't share alignment then we need to read to the
                                // intermediate buffer and then copy it to the correct location
                                uint32_t banked_alignment = banked_addr % ALIGNMENT;
                                if (dest_tr0_l1 % ALIGNMENT != banked_alignment) {
                                    // we write to the top of the intermediate buffer as that's aligned, and we write
                                    // from the closest align source address if source is not aligned to ALIGNMENT then
                                    // we go to the nearest address that is aligned and copy from there
                                    noc_async_read(banked_addr - (banked_alignment), intermed_l1_scratch, ALIGNMENT);
                                    volatile tt_l1_ptr uint8_t* dest_tr0_l1_ptr = (volatile uint8_t*)dest_tr0_l1;
                                    // need the barrier to ensure that we can copy from the intermediate buffer
                                    noc_async_read_barrier();
                                    // if source is not aligned to ALIGNMENT then we need to skip forward by the amount
                                    // needed to align to get to the correct data
                                    for (uint32_t i = 0; i < SUBTILE_LINE_BYTES; i++) {
                                        dest_tr0_l1_ptr[i] = intermed_l1_scratch_ptr[i + banked_alignment];
                                    }
                                } else {
                                    // if source and destination alignment are equivalent then we can read directly to
                                    // the destination
                                    noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);
                                }
                            } else {
                                noc_async_read(banked_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);
                            }
                            // Read the 16 elements for the row of the face directly adjacent since this comes from the
                            // same input tile
                            dest_tr0_l1 += 512;  // 16 subtile rows of 16 elements of 2 bytes for bfloat16 (16 * 16 * 2)
                            banked_addr += 512;
                        }

                        output_stick_id += output_Wt;
                    }
                    noc_async_read_barrier();
                    // notifies the unpacker that the buffer is populated
                    cb_push_back(cb_id_in0, onetile);

                    base_tile_stick_id += 1;
                }

                base_tile_row_stick_id += num_sticks_per_output_tile_row;
            }
        }
    }
}

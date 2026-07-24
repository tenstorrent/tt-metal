// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

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

    constexpr uint32_t ALIGNMENT = get_compile_time_arg_val(0);
    constexpr auto src0_args = TensorAccessorArgs<1>();

    uint32_t num_sticks_per_input_tile_row = input_Wt << 5;  // Tile height is 32
    uint32_t num_sticks_per_output_tile_row = output_Wt << 5;

    constexpr uint32_t SUBTILE_LINE_BYTES = (16 << 1);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dfb_id_in0 = 0;

    constexpr bool MISALIGNED = ALIGNMENT > SUBTILE_LINE_BYTES;
    constexpr uint32_t dfb_id_intermed = 1;
    DataflowBuffer dfb_intermed(dfb_id_intermed);
    uint32_t intermed_l1_scratch = MISALIGNED ? dfb_intermed.get_write_ptr() : 0;
    volatile tt_l1_ptr uint8_t* intermed_l1_scratch_ptr = (volatile uint8_t*)intermed_l1_scratch;

    const auto s0 = TensorAccessor(src0_args, src0_addr);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

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
                    dfb_in0.reserve_back(onetile);
                    for (uint32_t tile_h = 0; tile_h < 32; tile_h++) {
                        uint32_t input_tile_row_to_read = output_stick_id / num_sticks_per_input_tile_row;
                        uint32_t input_tile_col_to_read = output_stick_id % input_Wt;
                        uint32_t input_tile_to_read = input_tile_row_to_read * input_Wt + input_tile_col_to_read;
                        uint32_t input_tile_sub_row_to_read =
                            output_stick_id % num_sticks_per_input_tile_row / input_Wt;

                        // intra-tile offset within the source tile
                        uint32_t intra_tile_offset =
                            (((input_tile_sub_row_to_read >> 4) << 1) << 9) + ((input_tile_sub_row_to_read & 15) << 5);

                        uint32_t dest_tr0_l1 = dfb_in0.get_write_ptr();
                        dest_tr0_l1 +=
                            (((tile_h >> 4) << 1) << 9);  // if intra-tile source h is > 16, add 2*512 to subtile offset
                        dest_tr0_l1 += ((tile_h & 15) << 5);  // 16 * 2 bytes per face row

                        for (uint8_t i = 0; i < 2; ++i) {
                            if (MISALIGNED) {
                                // Need the absolute noc addr to compute alignment relative to source.
                                uint64_t banked_addr = s0.get_noc_addr(input_tile_to_read) + intra_tile_offset;
                                uint32_t banked_alignment = banked_addr % ALIGNMENT;
                                if (dest_tr0_l1 % ALIGNMENT != banked_alignment) {
                                    CoreLocalMem<uint32_t> scratch_dst(intermed_l1_scratch);
                                    noc.async_read(
                                        s0,
                                        scratch_dst,
                                        ALIGNMENT,
                                        {.page_id = input_tile_to_read,
                                         .offset_bytes = intra_tile_offset - banked_alignment},
                                        {.offset_bytes = 0});
                                    volatile tt_l1_ptr uint8_t* dest_tr0_l1_ptr = (volatile uint8_t*)dest_tr0_l1;
                                    noc.async_read_barrier();
                                    for (uint32_t i = 0; i < SUBTILE_LINE_BYTES; i++) {
                                        dest_tr0_l1_ptr[i] = intermed_l1_scratch_ptr[i + banked_alignment];
                                    }
                                } else {
                                    CoreLocalMem<uint32_t> dst(dest_tr0_l1);
                                    noc.async_read(
                                        s0,
                                        dst,
                                        SUBTILE_LINE_BYTES,
                                        {.page_id = input_tile_to_read, .offset_bytes = intra_tile_offset},
                                        {.offset_bytes = 0});
                                }
                            } else {
                                CoreLocalMem<uint32_t> dst(dest_tr0_l1);
                                noc.async_read(
                                    s0,
                                    dst,
                                    SUBTILE_LINE_BYTES,
                                    {.page_id = input_tile_to_read, .offset_bytes = intra_tile_offset},
                                    {.offset_bytes = 0});
                            }
                            // Read the 16 elements for the row of the face directly adjacent since this comes from the
                            // same input tile
                            dest_tr0_l1 += 512;  // 16 subtile rows of 16 elements of 2 bytes for bfloat16 (16 * 16 * 2)
                            intra_tile_offset += 512;
                        }

                        output_stick_id += output_Wt;
                    }
                    noc.async_read_barrier();
                    // notifies the unpacker that the buffer is populated
                    dfb_in0.push_back(onetile);

                    base_tile_stick_id += 1;
                }

                base_tile_row_stick_id += num_sticks_per_output_tile_row;
            }
        }
    }
}

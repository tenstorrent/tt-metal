// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using uint32_t = std::uint32_t;

// tile index to address
void kernel_main() {
    uint32_t input_Wt = get_arg(args::input_Wt);
    uint32_t output_N = get_arg(args::output_N);
    uint32_t output_C = get_arg(args::output_C);
    uint32_t output_Ht = get_arg(args::output_Ht);
    uint32_t output_Wt = get_arg(args::output_Wt);

    constexpr uint32_t ALIGNMENT = get_arg(args::ALIGNMENT);
    constexpr uint32_t ELEMENT_SIZE = get_arg(args::ELEMENT_SIZE);

    uint32_t num_sticks_per_input_tile_row = input_Wt << 5;  // Tile height is 32
    uint32_t num_sticks_per_output_tile_row = output_Wt << 5;

    // A tile is four 16x16 faces; a face row is 16 values and each tile row spans two faces.
    constexpr uint32_t SUBTILE_LINE_BYTES = 16 * ELEMENT_SIZE;
    constexpr uint32_t FACE_BYTES = 16 * SUBTILE_LINE_BYTES;
    constexpr uint32_t onetile = 1;

    constexpr bool MISALIGNED = ALIGNMENT > SUBTILE_LINE_BYTES;
    // The host declares the scratch DFB, and sets WIDE_ALIGNMENT_SCRATCH, only when the DRAM or L1 alignment is
    // wider than one face row of this element type. That is the same test as MISALIGNED above.
    // The host and the kernel spell the test separately. The asserts below fail the build if they disagree.
    // Without the define the token dfb::intermed does not exist. The handle then cannot be built.
    // A kernel that needs the scratch address but has no handle would copy through L1 address 0.
#ifdef WIDE_ALIGNMENT_SCRATCH
    static_assert(MISALIGNED, "host declared the scratch DFB but the kernel's own test disagrees");
    DataflowBuffer dfb_intermed(dfb::intermed);
    uint32_t intermed_l1_scratch = dfb_intermed.get_write_ptr();
#else
    static_assert(!MISALIGNED, "kernel needs the scratch DFB but the host did not declare it");
    uint32_t intermed_l1_scratch = 0;
#endif
    volatile tt_l1_ptr uint8_t* intermed_l1_scratch_ptr = (volatile uint8_t*)intermed_l1_scratch;

    const auto s0 = TensorAccessor(tensor::src0);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

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
                        uint32_t intra_tile_offset = (input_tile_sub_row_to_read >> 4) * 2 * FACE_BYTES +
                                                     (input_tile_sub_row_to_read & 15) * SUBTILE_LINE_BYTES;

                        uint32_t dest_tr0_l1 = dfb_in0.get_write_ptr();
                        dest_tr0_l1 += (tile_h >> 4) * 2 * FACE_BYTES;  // rows 16-31 are in the bottom two faces
                        dest_tr0_l1 += (tile_h & 15) * SUBTILE_LINE_BYTES;

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
                            dest_tr0_l1 += FACE_BYTES;
                            intra_tile_offset += FACE_BYTES;
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

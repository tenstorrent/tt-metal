// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "argmax_common.hpp"

#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "1. READER ARGMAX TILE LAYOUT: Starting kernel" << ENDL();

    // TODO: print addresses as hex

    // Compile time args
    // -----------------
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);

    constexpr uint32_t src_page_size = get_compile_time_arg_val(2) * sizeof(float);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(3) * sizeof(uint32_t);

    constexpr uint32_t tile_height = get_compile_time_arg_val(4);
    constexpr uint32_t tile_width = get_compile_time_arg_val(5);
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(6);
    constexpr uint32_t inner_dim_units = get_compile_time_arg_val(7);
    // This is the number of elements in the input tensor along the reduction dim (W)
    constexpr uint32_t reduce_units = get_compile_time_arg_val(7);
    constexpr bool reduce_all = (bool)get_compile_time_arg_val(7);

    DPRINT << "2. READER ARGMAX TILE LAYOUT: Compile time args" << ENDL();
    DPRINT << "src_cb_idx: " << src_cb_idx << ENDL();
    DPRINT << "dst_cb_idx: " << dst_cb_idx << ENDL();
    DPRINT << "src_page_size: " << src_page_size << ENDL();
    DPRINT << "dst_page_size: " << dst_page_size << ENDL();

    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);

    DPRINT << "3. READER ARGMAX TILE LAYOUT: Runtime args" << ENDL();
    DPRINT << "src_base_addr: " << src_base_addr << ENDL();
    DPRINT << "dst_base_addr: " << dst_base_addr << ENDL();

    // Tensor Accessors
    // ----------------
    constexpr auto s_src_args = TensorAccessorArgs<8>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();
    // Note, the page/tile sizes given in compile time args are used here
    // as a page size for accessing input, output tensors.
    const auto s_src = TensorAccessor(s_src_args, src_base_addr, src_page_size);
    const auto s_dst = TensorAccessor(s_dst_args, dst_base_addr, dst_page_size);

    // CB in L1 memory for storing input data.
    // We will work on 32x32 tiles of the input tensor, iterating 'left to right'.
    //
    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
    constexpr DataFormat src_cb_addr_data_format = get_dataformat(src_cb_idx);
    volatile tt_l1_ptr float* src_ptr = reinterpret_cast<volatile tt_l1_ptr float*>(src_cb_addr);

    // CB in L1 memory for storing output
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx);
    volatile tt_l1_ptr uint32_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);

    // Number of tiles (and pages) in 'height' dimension
    const int H = 1;
    // Number of tiles (and pages) in 'width' dimension
    const int W = 1;

    // TODO: How to obtain the size of a CB while in the kernel?
    //
    // 1024 elements in an output page
    // output tensor is in tile layout, tile dims: 32x32
    // logical dims (last two): 32x
    //
    // Initialize entire tensor page (including padding cells)
    // TODO: where to find the page size?
    for (int i = 0; i < 1024; i++) {
        src_ptr[i] = 10.123f;
        dst_ptr[i] = 17;
    }

    // TODO: what if input has entries == MIN_UINT32?
    // TODO: get a proper min inf
    float max_values[32] = {-1000000};
    uint32_t arg_max[32] = {0};

    // Iterate over the tiles of the input tensor.
    // TODO: does TensorAccessor have an iterator?
    // Loop over the rows of the input tensor (32 rows in one iteration)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            // Tiles of the tensor are indexed as row-major.
            const int tile_id = i * W + j;

            DPRINT << "Processing input Tile (" << i << ", " << j << ") " << "tile id: " << tile_id << ENDL();
            // Fetch input tile into cb0
            const uint64_t src_noc_addr = get_noc_addr(tile_id, s_src);
            // TODO: get the tile dimensions from somewhere.
            uint32_t read_size = 1024 * sizeof(float);
            noc_async_read(src_noc_addr, src_cb_addr, read_size);
            noc_async_read_barrier();

            // TODO: the tile size (in BH, WH) is most likely fixed to (32x32)
            // Process the tile of data (32x32)
            // Iterate over the (16x16) faces
            // TODO: can we use matrix/vector unit?
            // TODO: change k to f, because it is face index
            for (int k = 0; k < 4; k++) {
                // Get an offset to the face
                uint32_t face_offset = k * 16 * 16;
                // TODO: does tt_l1_ptr support pointer arithmetic?
                volatile tt_l1_ptr float* face_ptr = src_ptr + face_offset;

                // Go over the rows of the face. Update the maximum values in each row.
                for (int m = 0; m < 16; m++) {
                    // Row index in the tile
                    const uint32_t row_index = (k < 2) ? m : m + 16;
                    float curr_max = max_values[row_index];

                    if (row_index == 0) {
                        DPRINT << "Row=0 " << "Face=" << k << " max=" << curr_max << " index=" << arg_max[row_index]
                               << ENDL();
                        DPRINT << "Iterate over 16 row elements" << ENDL();
                    }

                    // Go over elements in the current row, current face
                    for (int n = 0; n < 16; n++) {
                        // TODO: Other data types
                        // Index within the face
                        uint32_t index = m * 16 + n;
                        float value = face_ptr[index];
                        if (row_index == 0) {
                            DPRINT << "\tn=" << n << " value=" << value << " max=" << curr_max << " index"
                                   << arg_max[row_index] << ENDL();
                        }
                        if (curr_max < value) {
                            // Element index in the row of th input tensor
                            // TODO: use row_index in the tensor (this works, for single tile tensors)
                            const bool is_left_side_face = (k == 0 || k == 2);
                            const uint32_t new_arg_max = j * 32 + (is_left_side_face ? 0 : 16) + n;

                            if (row_index == 0) {
                                DPRINT << "\tn=" << n << " new_max=" << value << " new_arg_max=" << new_arg_max
                                       << ENDL();
                            }
                            curr_max = value;
                            arg_max[row_index] = new_arg_max;
                        }
                    }
                    max_values[row_index] = curr_max;
                }
            }
        }

        // TODO: do this only if not reduce_all

        // Write to ouput the result of the current 32 rows
        // Logical shape is: [1x1x32x1]
        // Padded shape is: [1x1x32x32]
        // I.e., we want to write to the left-most column in a [32x32] matrix.
        //
        // Prepare tiled layout of the 32 argmax values that we have just prepared.
        // Store that layout in cb1.
        for (int idx = 0; idx < 32; idx++) {
            DPRINT << "ArgMax row=" << idx << " index=" << arg_max[idx] << ENDL();

            // In the output tensor, we write to the first column of the matrix
            // created by the last two dims. Currently, we assume that all dimensions
            // except the last two are =1.

            // Find the 16x16 (output) face, this element belongs to.
            // since we are in column 0, our output will go to one of the
            // 'left-side' faces.
            int face_id = (idx < 16) ? 0 : 2;
            // row within a 16x16 face
            int row_f = (face_id == 0) ? idx : idx - 16;

            // face is row-major, contiguous block of memory
            // find an index of the current element in this block
            // Note, we are in column 0 also within the face.
            int face_offset = row_f * 16;

            // find index (offset) within the 32x32 tile
            // first, find start offset of the face
            int tile_offset = face_id * (16 * 16);

            int index = tile_offset + face_offset;
            dst_ptr[index] = arg_max[idx];
        }

        // Copy contents of cb1 to the output tensor
        uint32_t out_tensor_tile_id = i;

        DPRINT << "Write to output tile id: " << i << ENDL();

        uint64_t dst_noc_addr = get_noc_addr(out_tensor_tile_id, s_dst);
        uint32_t write_size = 1024 * sizeof(uint32_t);
        noc_async_write(dst_cb_addr, dst_noc_addr, write_size);
        noc_async_write_barrier();

        // Initialize the buffer before the next round
        for (int idx = 0; idx < 32; idx++) {
            max_values[idx] = NEG_INF_FLOAT32;
        }
    }
}

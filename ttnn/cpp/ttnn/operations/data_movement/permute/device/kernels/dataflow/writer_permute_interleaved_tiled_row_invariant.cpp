// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "debug/dprint.h"

template <size_t N, typename T>
void dprint_array(T arr[N], const char* name) {
    DPRINT << name << ": ";
    for (size_t i = 0; i < N; i++) {
        DPRINT << arr[i] << " ";
    }
    DPRINT << ENDL();
}

// ------------------------------------------------------------------
// 1) unflatten_index<N>:
//    Unflatten 'flat_idx' in row-major order for a shape[] of length N.
//    shape[d] is also uint32_t. We store the result into out_multi_idx[].
template <uint32_t N>
inline void unflatten_index(uint32_t flat_idx, const uint32_t (&shape)[N], uint32_t (&out_multi_idx)[N]) {
    // Process from last dimension to first, in row-major unflattening.
    for (int d = N - 1; d >= 0; d--) {
        uint32_t dim_size = shape[d];
        out_multi_idx[d] = flat_idx % dim_size;
        flat_idx = flat_idx / dim_size;
    }
}

// ------------------------------------------------------------------
// 2) flatten_index_ignore_last_dim<N>:
//    Flatten the first (N-1) coords in row-major order, ignoring dimension N-1.
//    shape[] has length (N-1). The result is a uint32_t "row" offset.
template <uint32_t N>
inline uint32_t flatten_index_ignore_last_dim(const uint32_t (&multi_idx)[N - 1], const uint32_t (&shape)[N - 1]) {
    uint32_t offset = 0;
    for (uint32_t d = 0; d < N - 1; d++) {
        offset = offset * shape[d] + multi_idx[d];
    }
    return offset;
}

// ------------------------------------------------------------------
// 3) get_unpadded_linear_row_index_for_tile<N, TILE_HEIGHT, TILE_WIDTH>:
//    - 'tile' is a 0-based tile index in the "tiled_shape" (flattened row-major).
//    - 'input_tiled_shape':  length N, [ ..., X_t, W_t ]
//    - 'input_shape':        length N, [ ..., X,   W   ] (the unpadded shape).
//    - TILE_HEIGHT, TILE_WIDTH are compile-time constants.
//
//  Returns the linear row index (flattened in row-major ignoring the last dim)
//  in the UNPADDED shape where that tile starts.
//
//  Steps, conceptually:
//    a) unflatten 'tile' -> tile_multi_idx[N] in input_tiled_shape
//    b) x_t = tile_multi_idx[N-2]
//    c) x = x_t * TILE_HEIGHT
//    d) row_multi_idx[d < N-2] = tile_multi_idx[d], row_multi_idx[N-2] = x
//    e) flatten row_multi_idx[] vs input_shape[0..N-2]
//
template <uint32_t N, uint32_t TILE_HEIGHT, uint32_t TILE_WIDTH>
inline uint32_t get_unpadded_linear_row_index_for_tile(
    uint32_t tile,
    const uint32_t (&input_tiled_shape)[N],  // [ ..., X_t, W_t ]
    const uint32_t (&input_shape)[N]         // [ ..., X,   W   ]
) {
    // a) unflatten tile into tile_multi_idx
    uint32_t tile_multi_idx[N];
    unflatten_index<N>(tile, input_tiled_shape, tile_multi_idx);

    // b) x_t = tile_multi_idx[N-2]
    uint32_t x_t = tile_multi_idx[N - 2];

    // c) x = x_t * TILE_HEIGHT
    uint32_t x = x_t * TILE_HEIGHT;

    // d) Build row_multi_idx of length N-1, ignoring last dimension
    //    row_multi_idx[d] = tile_multi_idx[d] for d < N-2
    //    row_multi_idx[N-2] = x
    uint32_t row_multi_idx[N - 1];
    uint32_t row_shape[N - 1];
    for (uint32_t d = 0; d < N - 2; d++) {
        row_multi_idx[d] = tile_multi_idx[d];
        row_shape[d] = input_shape[d];
    }
    row_multi_idx[N - 2] = x;               // the actual row
    row_shape[N - 2] = input_shape[N - 2];  // dimension X

    // e) Flatten row_multi_idx in row_shape => linear row index
    return flatten_index_ignore_last_dim<N>(row_multi_idx, row_shape);
}

void kernel_main() {
    // Compile-time constants
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t element_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(2);
    constexpr uint32_t X = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(6);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(7);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(8);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(9);
    constexpr bool needs_padding = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t N = get_compile_time_arg_val(11);

    DPRINT << "Starting writer_permute_interleaved_tiled_row_invariant" << ENDL();
    DPRINT << "dst_is_dram: " << (uint32_t)dst_is_dram << ENDL();
    DPRINT << "element_size: " << element_size << ENDL();
    DPRINT << "cb_id_out0: " << cb_id_out0 << ENDL();
    DPRINT << "X: " << X << ENDL();
    DPRINT << "H: " << H << ENDL();
    DPRINT << "W: " << W << ENDL();
    DPRINT << "TILE_HEIGHT: " << TILE_HEIGHT << ENDL();
    DPRINT << "TILE_WIDTH: " << TILE_WIDTH << ENDL();
    DPRINT << "FACE_HEIGHT: " << FACE_HEIGHT << ENDL();
    DPRINT << "FACE_WIDTH: " << FACE_WIDTH << ENDL();
    DPRINT << "needs_padding: " << (uint32_t)needs_padding << ENDL();
    DPRINT << "N: " << N << ENDL();

    // Retrieve arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile = get_arg_val<uint32_t>(1);
    uint32_t end_tile = get_arg_val<uint32_t>(2);
    uint32_t start_padding_tile_idx = get_arg_val<uint32_t>(3);
    uint32_t end_padding_tile_idx = get_arg_val<uint32_t>(4);

    DPRINT << "dst_addr: " << dst_addr << ENDL();
    DPRINT << "start_tile: " << start_tile << ENDL();
    DPRINT << "end_tile: " << end_tile << ENDL();
    DPRINT << "start_padding_tile_idx: " << start_padding_tile_idx << ENDL();
    DPRINT << "end_padding_tile_idx: " << end_padding_tile_idx << ENDL();

    // Input shape, permutation, and destination strides
    uint32_t array_start_offset = 5;  // input shape starts at arg 5
    uint32_t input_shape[N], perm[N], output_shape[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + array_start_offset);
        perm[i] = get_arg_val<uint32_t>(i + array_start_offset + N);
    }
    for (uint32_t i = 0; i < N; i++) {
        output_shape[i] = input_shape[perm[i]];
    }
    dprint_array<N, uint32_t>(input_shape, "input_shape");
    dprint_array<N, uint32_t>(perm, "perm");

    // Derived compile-time constants
    constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
    constexpr uint8_t NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT;
    constexpr uint8_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;

    constexpr uint32_t X_p = tt::data_movement::common::round_up<X, TILE_HEIGHT>();
    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    constexpr uint32_t W_t = W_p / TILE_WIDTH;
    constexpr uint32_t H_t = H_p / TILE_HEIGHT;
    constexpr uint32_t X_t = X_p / TILE_HEIGHT;

    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;

    DPRINT << "TILE_HW: " << TILE_HW << ENDL();
    DPRINT << "NUM_FACES_H: " << (uint32_t)NUM_FACES_H << ENDL();
    DPRINT << "NUM_FACES_W: " << (uint32_t)NUM_FACES_W << ENDL();
    DPRINT << "X_p: " << X_p << ENDL();
    DPRINT << "H_p: " << H_p << ENDL();
    DPRINT << "W_p: " << W_p << ENDL();
    DPRINT << "W_t: " << W_t << ENDL();
    DPRINT << "H_t: " << H_t << ENDL();
    DPRINT << "X_t: " << X_t << ENDL();

    // Initialize address generator
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const auto input_data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram, TILE_HW> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = input_data_format};

    // Calculate actual data height in the last tile
    constexpr uint32_t H_last_tile = H - (H_t - 1) * TILE_HEIGHT;

    // Calculate real_faces_h
    uint8_t remainder_faces_h = tt::data_movement::common::div_up<H_last_tile, FACE_HEIGHT>();

    uint32_t remainder = H_last_tile % FACE_HEIGHT;
    uint8_t sub_tile_lines_real = (remainder == 0) ? FACE_HEIGHT : static_cast<uint8_t>(remainder);

    // Precompute constants used in inner loops
    const uint32_t face_height_width = FACE_HEIGHT * FACE_WIDTH;
    const uint32_t num_faces_wh = NUM_FACES_W * FACE_WIDTH;

    // Main single loop over all tiles
    uint32_t src_multi_idx[N];
    uint32_t dest_multi_idx[N];

    uint32_t input_padded_shape[N];
    for (uint32_t i = 0; i < N - 2; i++) {
        input_padded_shape[i] = input_shape[i];
    }
    input_padded_shape[N - 2] = H_p;
    input_padded_shape[N - 1] = W_p;
    dprint_array<N, uint32_t>(input_padded_shape, "input_padded_shape");

    uint32_t input_tiled_shape[N];
    for (uint32_t i = 0; i < N - 2; i++) {
        input_tiled_shape[i] = input_padded_shape[i];
    }
    input_tiled_shape[N - 2] = H_t;
    input_tiled_shape[N - 1] = W_t;
    dprint_array<N, uint32_t>(input_tiled_shape, "input_tiled_shape");

    uint32_t output_padded_shape[N];
    for (uint32_t i = 0; i < N - 2; i++) {
        output_padded_shape[i] = output_shape[i];
    }
    output_padded_shape[N - 2] = X_p;
    output_padded_shape[N - 1] = W_p;
    dprint_array<N, uint32_t>(output_padded_shape, "output_padded_shape");

    uint32_t output_tiled_shape[N];
    for (uint32_t i = 0; i < N - 2; i++) {
        output_tiled_shape[i] = output_padded_shape[i];
    }
    output_tiled_shape[N - 2] = X_t;
    output_tiled_shape[N - 1] = W_t;
    dprint_array<N, uint32_t>(output_tiled_shape, "output_tiled_shape");
    DPRINT << ENDL();

    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        uint32_t tile_start =
            get_unpadded_linear_row_index_for_tile<N, TILE_HEIGHT, TILE_WIDTH>(tile, input_tiled_shape, input_shape);
        uint32_t w_t = tile % W_t;
        uint32_t temp = tile / W_t;
        uint32_t h_t = temp % H_t;

        // Determine the number of faces in the height dimension
        uint8_t num_faces_h = (h_t == H_t - 1) ? remainder_faces_h : NUM_FACES_H;
        cb_wait_front(cb_id_out0, 1);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        // DPRINT << "tile_start: " << tile_start << ENDL();
        // Iterate over faces in the height dimension
        for (uint8_t face_h = 0; face_h < num_faces_h; ++face_h) {
            // Determine the number of sub-tile lines to process
            bool is_last_sub_tile_line = (h_t == H_t - 1) && (face_h == num_faces_h - 1);
            uint8_t sub_tile_lines = is_last_sub_tile_line ? sub_tile_lines_real : FACE_HEIGHT;

            // Iterate over faces in the width dimension
            for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                // Iterate over sub-tile lines
                for (uint8_t sub_tile_line = 0; sub_tile_line < sub_tile_lines; ++sub_tile_line) {
                    // Compute multi-dimensional index for the row that we're on
                    uint32_t row = tile_start + (uint32_t)(face_h * FACE_HEIGHT + sub_tile_line);
                    uint32_t original_row = row;
                    for (uint32_t i = 0; i < N - 1; ++i) {
                        size_t dim = N - 2 - i;  // Start from the second last dimension
                        src_multi_idx[dim] = row % input_shape[dim];
                        row /= input_shape[dim];
                    }
                    src_multi_idx[N - 1] = 0;  // Logical row dimension index for output tensor

                    // Apply permutation to get destination multi-dimensional index
                    for (uint32_t i = 0; i < N; ++i) {
                        dest_multi_idx[i] = src_multi_idx[perm[i]];  // Logical row dimension index for output tensor
                    }
                    // Convert destination multi-dimensional index to linear index
                    // Account for tiled, faced/subtiled shape of the destination tensor
                    // tensor is permuted from input: [..., X, ... H, ...W] to output: [..., H, ...X, W]
                    // tensors are tiled as input: [..., X, ... H/TILE_HEIGHT, W/TILE_WIDTH] and output: [..., H,
                    // ...X/TILE_HEIGHT, W/TILE_WIDTH] each tensor tile is faced as input: [..., X, ... H/TILE_HEIGHT,
                    // W/TILE_WIDTH, NUM_FACES_H, NUM_FACES_W, FACE_HEIGHT, FACE_WIDTH] and output: [..., H,
                    // ...X/TILE_HEIGHT, W/TILE_WIDTH, NUM_FACES_H, NUM_FACES_W, FACE_HEIGHT, FACE_WIDTH] where
                    // NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT and NUM_FACES_W = TILE_WIDTH / FACE_WIDTH

                    // First find the tile that this belongs to
                    // 1) Flatten all outer dimensions into outer_flat
                    // logical output shape: [..., X, W]
                    // padded output shape: [..., X_p, W_p]
                    // tiled output shape: [..., X_t, W_t] = [..., X_p/TILE_HEIGHT, W_p/TILE_WIDTH]
                    uint32_t output_row_offset = 0;
                    for (uint32_t i = 0; i < N - 1; i++) {
                        output_row_offset = output_row_offset * output_padded_shape[i] + dest_multi_idx[i];
                    }

                    uint32_t output_tile_idx = (output_row_offset / TILE_HEIGHT) * W_t + w_t;

                    // 1) The row coordinate in the X dimension:
                    uint32_t output_row = dest_multi_idx[N - 2];

                    // 2) Position of this output_row within the tile:
                    uint32_t output_row_in_tile = output_row % TILE_HEIGHT;

                    // 3) Face index along the tile's height dimension:
                    uint32_t output_face_h = output_row_in_tile / FACE_HEIGHT;  // in [0..NUM_FACES_H-1]

                    // 4) Row within that face:
                    uint32_t output_sub_tile_line = output_row_in_tile % FACE_HEIGHT;  // in [0..FACE_HEIGHT-1]

                    // Precompute offset for the current face_h
                    uint32_t face_h_offset = output_face_h * NUM_FACES_W * face_height_width;

                    // Combine face_w for the overall tile column
                    uint32_t face_w_offset = face_w * face_height_width;

                    // Since we are writing in SUBTILE_LINE_BYTES chunks, we don't need an offset within the sub-tile
                    // line

                    // Sub-tile/face line offset where our data starts in the tile
                    uint32_t offset =
                        (face_h_offset + face_w_offset + output_sub_tile_line * FACE_WIDTH) * element_size;

                    uint64_t write_noc_base_addr = get_noc_addr(output_tile_idx, s, offset);
                    {
                        // DPRINT << "input tile: " << tile << ENDL();
                        // DPRINT << "starting linear index of tile: " << tile_start << ENDL();
                        // DPRINT << "linear input row: " << original_row << ENDL();
                        // dprint_array<N, uint32_t>(src_multi_idx, "source index");
                        // dprint_array<N, uint32_t>(dest_multi_idx, "dest index");
                        // DPRINT << "sub_tile_line: " << (uint32_t) sub_tile_line << ENDL();

                        // DPRINT << "output_row_offset: " << output_row_offset << ENDL();
                        // DPRINT << "output_tile_idx: " << output_tile_idx << ENDL();
                        // DPRINT << "output_row in tile: " << output_row_in_tile << ENDL();
                        // DPRINT << "output_face_h: " << output_face_h << ENDL();
                        // DPRINT << "output_face_w: " << (uint32_t) face_w << ENDL();
                        // DPRINT << "output_sub_tile_line: " << output_sub_tile_line << ENDL();
                        // DPRINT << "face_h_offset: " << face_h_offset << ENDL();
                        // DPRINT << "face_w_offset: " << face_w_offset << ENDL();
                        // DPRINT << "offset: " << offset << ENDL();
                        // // DPRINT << "write_noc_base_addr: " << write_noc_base_addr << ENDL();
                        // tt_l1_ptr uint16_t* l1_ptr = reinterpret_cast<tt_l1_ptr uint16_t*>(l1_read_addr);
                        // DPRINT << "l1_ptr[0]: " << BF16(l1_ptr[0]) << ENDL();
                        // DPRINT << ENDL();
                    }
                    // Perform asynchronous write
                    noc_async_write(l1_read_addr, write_noc_base_addr, SUBTILE_LINE_BYTES);

                    // Increment the read address
                    l1_read_addr += SUBTILE_LINE_BYTES;
                }

                // Skip padding if not all lines are real
                if (is_last_sub_tile_line) {
                    l1_read_addr += (FACE_HEIGHT - sub_tile_lines) * SUBTILE_LINE_BYTES;
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }

    // add padding
    if constexpr (needs_padding) {
        DPRINT << "Adding padding" << ENDL();
        cb_wait_front(tt::CBIndex::c_1, 1);
        uint32_t l1_read_addr = get_read_ptr(tt::CBIndex::c_1);
        tt_l1_ptr uint16_t* l1_ptr = reinterpret_cast<tt_l1_ptr uint16_t*>(l1_read_addr);
        DPRINT << "l1_ptr[0]: " << BF16(l1_ptr[0]) << ENDL();

        uint32_t l1_read_ptr = get_read_ptr(tt::CBIndex::c_1);

        constexpr uint32_t x_t = X_t - 1;
        constexpr uint8_t X_in_tile = X % TILE_HEIGHT;
        constexpr uint8_t face_c_start = X_in_tile / FACE_HEIGHT;

        for (uint32_t tile_idx = start_padding_tile_idx; tile_idx < end_padding_tile_idx; ++tile_idx) {
            // Map tile_idx to (n, h, w_t)

            // input tiled shape is [...., X_t, W_t],
            // we want to iterate across:
            // [0, ...., X_t-1, 0] to [N-1, ...., Xt-1, W_t-1]
            // tile_idx is between [0, prod(N, ..., Xt, W_t)/Xt)

            // calculate dest_multi_idx
            size_t remaining = tile_idx;
            for (uint32_t i = 0; i < N; ++i) {
                size_t dim = N - 1 - i;
                if (dim == N - 2) {
                    dest_multi_idx[dim] = x_t;
                    continue;
                }
                dest_multi_idx[dim] = remaining % output_tiled_shape[dim];
                remaining /= output_tiled_shape[dim];
            }
            dprint_array<N, uint32_t>(dest_multi_idx, "dest_multi_idx");

            // Calculate linear_idx of padded tile inside output tensor buffer
            uint32_t linear_idx = 0;
            for (uint32_t i = 0; i < N; ++i) {
                linear_idx = (linear_idx * output_tiled_shape[i]) + dest_multi_idx[i];
            }
            DPRINT << "linear_idx: " << linear_idx << ENDL();

            for (uint8_t face_c = face_c_start; face_c < NUM_FACES_H; ++face_c) {
                // Offset to the start of the current face along the channel dimension/height of the tile
                uint32_t face_c_offset = face_c * NUM_FACES_W * face_height_width;

                // Sub-tile/face line where our padded data starts
                uint8_t sub_tile_line_start = face_c == face_c_start ? X_in_tile % FACE_HEIGHT : 0;

                for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                    // Offset to the start of the current face along the width of the tile
                    uint32_t face_w_offset = face_w * face_height_width;
                    for (uint8_t sub_tile_line = sub_tile_line_start; sub_tile_line < FACE_HEIGHT; ++sub_tile_line) {
                        // offset to the start of the current sub-tile line
                        uint32_t offset = (face_c_offset + face_w_offset + sub_tile_line * FACE_WIDTH) * element_size;

                        // Compute the write address
                        uint64_t write_noc_base_addr = get_noc_addr(linear_idx, s, offset);

                        // Perform asynchronous write
                        noc_async_write(l1_read_ptr, write_noc_base_addr, SUBTILE_LINE_BYTES);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(tt::CBIndex::c_1, 1);
    }
}

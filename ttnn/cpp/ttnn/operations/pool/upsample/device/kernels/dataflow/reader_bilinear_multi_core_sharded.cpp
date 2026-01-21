// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "fixed_point_arithmetic.hpp"
#include "bilinear_weights_lut.hpp"

//
// Halo padding configuration
//
// The input tensor is padded with a 1-pixel border (halo) on all sides to handle
// boundary conditions during bilinear interpolation. This allows edge pixels to be
// interpolated correctly without special-case logic.
//
constexpr uint32_t PADDING_TOP = 1;
constexpr uint32_t PADDING_BOTTOM = 1;
constexpr uint32_t PADDING_LEFT = 1;
constexpr uint32_t PADDING_RIGHT = 1;
constexpr uint32_t TOTAL_HEIGHT_PADDING = PADDING_TOP + PADDING_BOTTOM;
constexpr uint32_t TOTAL_WIDTH_PADDING = PADDING_LEFT + PADDING_RIGHT;

//
// Packs four BF16 weights into two uint32_t values and writes them to L1 memory
//
// Layout: [w1, w2] in first uint32, [w3, w4] in second uint32
//         Each weight occupies 16 bits in BF16 format
//
// These weights correspond to the 4 corners of the bilinear interpolation:
//   w1 (val):  top-left corner weight
//   w2 (val1): top-right corner weight
//   w3 (val2): bottom-left corner weight
//   w4 (val3): bottom-right corner weight
//
ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

//
// BilinearIndexAdvancer: Tracks state for bilinear interpolation during upsampling
//
// Purpose:
//   Manages the iteration through output pixels, computing the corresponding input
//   coordinates and bilinear weights for each output position. Handles wraparound
//   at row/batch boundaries and computes addresses for the 4 input neighbors needed
//   for each output pixel.

// Explanation:
// Advancement is done in row-major order. For each output pixel, the advancer
// calculates the fixed-point input coordinates, determines the 4 neighboring input
// pixels (with boundary clamping), and retrieves the pre-computed bilinear weights
// A phase is defined as the position within the scale factor grid, which is used to look up
// the appropriate weights from a LUT.
// There is scale_h * scale_w unique phases, each with its own set of weights.

//
// Template Parameters:
//   OUT_H, OUT_W: Output dimensions
//   IN_H, IN_W: Input dimensions
//   SCALE_H_INV, SCALE_W_INV: Inverse of scale factors (in fixed-point)
//   X/Y_STARTING_COORDINATE: Initial coordinates for first output pixel (fixed-point)
//   STICK_NBYTES: Bytes per row of data
//   SCALE_H, SCALE_W: Upsampling scale factors (used for phase calculation)
//
template <
    uint32_t OUT_H,
    uint32_t OUT_W,
    uint32_t IN_H,
    uint32_t IN_W,
    int32_t SCALE_H_INV,
    int32_t SCALE_W_INV,
    int32_t X_STARTING_COORDINATE,
    int32_t Y_STARTING_COORDINATE_FIXED,
    uint32_t STICK_NBYTES,
    uint32_t SCALE_H,
    uint32_t SCALE_W>
struct BilinearIndexAdvancer {
    static constexpr uint32_t HALO_PADDED_WIDTH = IN_W + TOTAL_WIDTH_PADDING;
    static constexpr uint32_t HALO_PADDED_HEIGHT = IN_H + TOTAL_HEIGHT_PADDING;

    uint32_t current_output_row;
    uint32_t current_output_col;
    uint32_t batch_id;
    uint32_t starting_batch_id;

    int32_t y_coordinate;
    int32_t x_coordinate;

    uint32_t l1_read_addr;

    uint32_t halo_starting_row_offset;
    uint32_t halo_starting_col_offset;

    uint32_t phase_h;
    uint32_t phase_w;

    //
    // Initializes advancer state for a given starting output pixel
    //
    // start_output_idx: Flattened index of first output pixel this core produces
    // min_input_offset: Starting offset in the halo-padded input buffer
    // l1_read_addr_: Base L1 address of the input halo buffer
    //
    ALWI BilinearIndexAdvancer(uint32_t start_output_idx, uint32_t min_input_offset, uint32_t l1_read_addr_) :
        l1_read_addr(l1_read_addr_) {
        batch_id = start_output_idx / (OUT_H * OUT_W);
        uint32_t in_batch_idx = start_output_idx % (OUT_H * OUT_W);
        current_output_row = in_batch_idx / OUT_W;
        current_output_col = in_batch_idx % OUT_W;

        y_coordinate = Y_STARTING_COORDINATE_FIXED + (current_output_row * SCALE_H_INV);
        x_coordinate = X_STARTING_COORDINATE + (current_output_col * SCALE_W_INV);

        uint32_t min_batch_id = min_input_offset / (HALO_PADDED_HEIGHT * HALO_PADDED_WIDTH);
        uint32_t min_in_batch_idx = min_input_offset % (HALO_PADDED_HEIGHT * HALO_PADDED_WIDTH);
        halo_starting_row_offset = min_in_batch_idx / HALO_PADDED_WIDTH;
        halo_starting_col_offset = min_in_batch_idx % HALO_PADDED_WIDTH;

        starting_batch_id = min_batch_id;

        phase_h = current_output_row % SCALE_H;
        phase_w = current_output_col % SCALE_W;
    }

    //
    // Advances to the next output pixel (row-major order)
    // Handles wraparound at row and batch boundaries
    //
    ALWI void advance() {
        current_output_col++;
        x_coordinate += SCALE_W_INV;
        phase_w = (phase_w == SCALE_W - 1) ? 0 : phase_w + 1;

        if (current_output_col >= OUT_W) {
            current_output_col = 0;
            current_output_row++;

            y_coordinate += SCALE_H_INV;
            x_coordinate = X_STARTING_COORDINATE;
            phase_w = 0;
            phase_h = (phase_h == SCALE_H - 1) ? 0 : phase_h + 1;

            if (current_output_row == OUT_H) {
                batch_id++;
                current_output_row = 0;
                phase_h = 0;

                y_coordinate = Y_STARTING_COORDINATE_FIXED;
            }
        }
    }

    //
    // Computes bilinear interpolation data for the current output pixel
    //
    // Returns:
    //   - L1 addresses for the 4 input neighbors (y1x1, y1x2, y2x1, y2x2)
    //   - BF16 weights for each neighbor (p1, p2, p3, p4)
    //
    // Handles:
    //   - Boundary clamping: prevents sampling outside input tensor bounds
    //   - Weight lookup: retrieves pre-computed weights from LUT based on phase
    //   - Address calculation: computes L1 addresses relative to halo buffer start
    //
    ALWI void get_bilinear_data(
        uint32_t& y1x1_addr,
        uint32_t& y1x2_addr,
        uint32_t& y2x1_addr,
        uint32_t& y2x2_addr,
        uint16_t& weight_top_left_bf16,
        uint16_t& weight_top_right_bf16,
        uint16_t& weight_bottom_left_bf16,
        uint16_t& weight_bottom_right_bf16) const {
        // Convert fixed-point coordinates to integer pixel positions
        uint32_t y1_raw = fixed_point_arithmetic::fixed_to_int(y_coordinate);
        uint32_t x1_raw = fixed_point_arithmetic::fixed_to_int(x_coordinate);

        // Compute the 4 neighbor positions with boundary clamping
        // The halo padding means actual data starts at index 1, not 0
        uint32_t y1 = y1_raw;
        uint32_t y2 = y1_raw + 1;

        if (y1_raw == 0) {
            y1 = 1;
        }
        if (y2 > IN_H) {
            y2 = IN_H;
        }

        uint32_t x1 = x1_raw;
        uint32_t x2 = x1_raw + 1;

        if (x1_raw == 0) {
            x1 = 1;
        }
        if (x2 > IN_W) {
            x2 = IN_W;
        }

        // Lookup pre-computed weights from LUT based on current phase
        uint32_t lut_idx = (phase_h * SCALE_W + phase_w) * 2;
        uint32_t packed_top_left_top_right = BilinearWeightsLUT<SCALE_H, SCALE_W>::weights.data[lut_idx];
        uint32_t packed_bottom_left_bottom_right = BilinearWeightsLUT<SCALE_H, SCALE_W>::weights.data[lut_idx + 1];

        // Unpack BF16 weights (each uint32 contains 2 BF16 values)
        weight_top_left_bf16 = packed_top_left_top_right & 0xFFFF;
        weight_top_right_bf16 = packed_top_left_top_right >> 16;
        weight_bottom_left_bf16 = packed_bottom_left_bottom_right & 0xFFFF;
        weight_bottom_right_bf16 = packed_bottom_left_bottom_right >> 16;

        // Calculate addresses in halo-padded input buffer
        int32_t batch_diff = batch_id - starting_batch_id;

        int32_t rel_y1 = static_cast<int32_t>(y1) - static_cast<int32_t>(halo_starting_row_offset);
        int32_t rel_y2 = static_cast<int32_t>(y2) - static_cast<int32_t>(halo_starting_row_offset);
        int32_t rel_x1 = static_cast<int32_t>(x1) - static_cast<int32_t>(halo_starting_col_offset);
        int32_t rel_x2 = static_cast<int32_t>(x2) - static_cast<int32_t>(halo_starting_col_offset);

        uint32_t stick_idx_y1x1 = batch_diff * (HALO_PADDED_HEIGHT * HALO_PADDED_WIDTH) +
                                  static_cast<uint32_t>(rel_y1) * HALO_PADDED_WIDTH + static_cast<uint32_t>(rel_x1);
        uint32_t stick_idx_y1x2 = batch_diff * (HALO_PADDED_HEIGHT * HALO_PADDED_WIDTH) +
                                  static_cast<uint32_t>(rel_y1) * HALO_PADDED_WIDTH + static_cast<uint32_t>(rel_x2);
        uint32_t stick_idx_y2x1 = batch_diff * (HALO_PADDED_HEIGHT * HALO_PADDED_WIDTH) +
                                  static_cast<uint32_t>(rel_y2) * HALO_PADDED_WIDTH + static_cast<uint32_t>(rel_x1);
        uint32_t stick_idx_y2x2 = batch_diff * (HALO_PADDED_HEIGHT * HALO_PADDED_WIDTH) +
                                  static_cast<uint32_t>(rel_y2) * HALO_PADDED_WIDTH + static_cast<uint32_t>(rel_x2);

        y1x1_addr = l1_read_addr + stick_idx_y1x1 * STICK_NBYTES;
        y1x2_addr = l1_read_addr + stick_idx_y1x2 * STICK_NBYTES;
        y2x1_addr = l1_read_addr + stick_idx_y2x1 * STICK_NBYTES;
        y2x2_addr = l1_read_addr + stick_idx_y2x2 * STICK_NBYTES;
    }
};

//
// Kernel entry point: Bilinear upsampling reader (multi-core sharded version)
//
// This kernel reads halo-padded input data from L1 and produces output pixels through
// bilinear interpolation. Each core handles a subset of output pixels. For efficiency,
// each core runs two threads (reader/writer) that alternate producing pixels.
//
// Architecture:
//   - Input: Halo-padded tensor in L1 circular buffer (halo_cb)
//   - Output: Interpolation data written to tilize_reduce_cb (4 neighbors per pixel)
//            + weights written to in_scalar_cb (4 BF16 weights per pixel)
//   - Downstream: Compute kernel performs weighted reduction
//
// Data Flow (per output pixel):
//   1. Advancer computes 4 neighbor addresses + weights from LUT
//   2. For each channel block:
//      - Read 4 neighbor stick segments via NOC
//      - Write weights to scalar CB
//      - Push data to compute kernel
//   3. Advance to next pixel (skipping one for reader/writer split)
//
void kernel_main() {
    //
    // Runtime arguments
    //
    uint32_t start_output_idx = get_arg_val<uint32_t>(0);
    uint32_t min_input_offset = get_arg_val<uint32_t>(1);
    uint32_t output_shard_height = get_arg_val<uint32_t>(2);

    //
    // Compile-time arguments: Tensor dimensions and scaling
    //
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t scale_h = get_compile_time_arg_val(1);
    constexpr uint32_t scale_w = get_compile_time_arg_val(2);
    constexpr uint32_t in_w = get_compile_time_arg_val(3);
    constexpr uint32_t out_w = get_compile_time_arg_val(4);
    constexpr uint32_t in_h = get_compile_time_arg_val(5);

    //
    // Compile-time arguments: Circular buffer IDs
    //
    constexpr uint32_t halo_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t tilize_reduce_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(8);

    //
    // Compile-time arguments: Fixed-point coordinate transforms
    //
    constexpr uint32_t scale_h_inv_fixed_u32 = get_compile_time_arg_val(9);
    constexpr uint32_t scale_w_inv_fixed_u32 = get_compile_time_arg_val(10);
    constexpr uint32_t y_starting_coordinate_fixed_u32 = get_compile_time_arg_val(11);
    constexpr uint32_t x_starting_coordinate_fixed_u32 = get_compile_time_arg_val(12);

    //
    // Compile-time arguments: Threading and blocking
    //
    constexpr uint32_t is_reader = get_compile_time_arg_val(13);
    constexpr uint32_t blocks = get_compile_time_arg_val(14);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(15);

    uint32_t l1_read_addr = get_read_ptr(halo_cb_id);

    // Split work between reader and writer threads
    // Reader gets ceiling(N/2), writer gets floor(N/2)
    uint32_t output_pixels_per_core = is_reader ? (output_shard_height + 1) / 2 : output_shard_height / 2;

    constexpr uint32_t out_h = in_h * scale_h;

    // Cast fixed-point values from host
    constexpr int32_t scale_h_inv = static_cast<int32_t>(scale_h_inv_fixed_u32);
    constexpr int32_t scale_w_inv = static_cast<int32_t>(scale_w_inv_fixed_u32);
    constexpr int32_t y_starting_coordinate_fixed = static_cast<int32_t>(y_starting_coordinate_fixed_u32);
    constexpr int32_t x_starting_coordinate_fixed = static_cast<int32_t>(x_starting_coordinate_fixed_u32);

    // Initialize the advancer to track our position through output space
    BilinearIndexAdvancer<
        out_h,
        out_w,
        in_h,
        in_w,
        scale_h_inv,
        scale_w_inv,
        x_starting_coordinate_fixed,
        y_starting_coordinate_fixed,
        stick_nbytes,
        scale_h,
        scale_w>
        advancer(start_output_idx, min_input_offset, l1_read_addr);

    // Writer starts one pixel ahead (interleaving pattern)
    if constexpr (!is_reader) {
        advancer.advance();
    }

    //
    // Main loop: Process assigned output pixels
    //
    for (uint32_t output_pixel = 0; output_pixel < output_pixels_per_core; ++output_pixel) {
        // Get interpolation data for current output pixel
        uint32_t y1x1_addr, y1x2_addr, y2x1_addr, y2x2_addr;
        uint16_t weight_top_left_bf16, weight_top_right_bf16, weight_bottom_left_bf16, weight_bottom_right_bf16;
        advancer.get_bilinear_data(
            y1x1_addr,
            y1x2_addr,
            y2x1_addr,
            y2x2_addr,
            weight_top_left_bf16,
            weight_top_right_bf16,
            weight_bottom_left_bf16,
            weight_bottom_right_bf16);

        constexpr uint32_t last_block_size_bytes = stick_nbytes - (blocks - 1) * input_block_size_bytes;

        // Process each channel block
        uint32_t block_offset = 0;
#pragma unroll
        for (uint32_t i = 0; i < blocks; i++) {
            cb_reserve_back(tilize_reduce_cb_id, 4);

            uint32_t current_block_size_bytes = (i == blocks - 1) ? last_block_size_bytes : input_block_size_bytes;

            uint32_t l1_write_addr = get_write_ptr(tilize_reduce_cb_id);

            // Read 4 neighbor stick segments
            noc_async_read(get_noc_addr(y1x1_addr + block_offset), l1_write_addr, current_block_size_bytes);
            l1_write_addr += input_block_size_bytes;

            noc_async_read(get_noc_addr(y1x2_addr + block_offset), l1_write_addr, current_block_size_bytes);
            l1_write_addr += input_block_size_bytes;

            noc_async_read(get_noc_addr(y2x1_addr + block_offset), l1_write_addr, current_block_size_bytes);
            l1_write_addr += input_block_size_bytes;

            noc_async_read(get_noc_addr(y2x2_addr + block_offset), l1_write_addr, current_block_size_bytes);

            // Write weights for compute kernel
            fill_four_val(
                get_write_ptr(in_scalar_cb_id),
                weight_top_left_bf16,
                weight_top_right_bf16,
                weight_bottom_left_bf16,
                weight_bottom_right_bf16);
            cb_push_back(in_scalar_cb_id, 1);

            cb_push_back(tilize_reduce_cb_id, 4);
            block_offset += current_block_size_bytes;
        }

        // Advance twice: skip next pixel (handled by other thread)
        advancer.advance();
        advancer.advance();
    }
}

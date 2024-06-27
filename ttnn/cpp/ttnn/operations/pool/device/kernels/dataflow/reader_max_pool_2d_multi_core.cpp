// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// #include "debug/dprint.h"

// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1 };

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

inline bool fill_with_val_async(const InterleavedPow2AddrGenFast<false>& s_const, uint32_t begin_addr, int32_t nrows, uint32_t row_nbytes) {
    uint32_t curr_addr = begin_addr;
    for (int32_t row_i = 0; row_i < nrows; ++ row_i) {
        s_const.noc_async_read_page(0, curr_addr);
        curr_addr += row_nbytes;
    }
    return true;
}

/**
 * Max-pool 2D.
 */
void kernel_main() {
    // input tensor address
    const uint32_t in_addr = get_arg_val<uint32_t>(0);

    // max pool window size height / width
    const uint32_t window_h = get_arg_val<uint32_t>(2);
    const uint32_t window_w = get_arg_val<uint32_t>(3);
    // product of window_h and window_w
    const int32_t window_hw = get_arg_val<int32_t>(4);
    // window_hw_padded = window_hw rounded up to the tile size (can be multiple tiles)
    const uint32_t window_hw_padded = get_arg_val<uint32_t>(5);

    // max pool padding height / width
    const int32_t pad_h = get_arg_val<int32_t>(8);
    const int32_t pad_w = get_arg_val<int32_t>(9);

    // output tensor height / width
    const int32_t out_h = get_arg_val<int32_t>(10);
    const int32_t out_w = get_arg_val<int32_t>(11);

    // channel size in bytes, multiple of 32
    const uint32_t in_nbytes_c = get_arg_val<uint32_t>(14);

    // input tensor height / width / channels
    const int32_t in_h = get_arg_val<int32_t>(16);
    const int32_t in_w = get_arg_val<int32_t>(17);
    const int32_t in_c = get_arg_val<int32_t>(19);

    // input CB page szie
    const int32_t in_cb_pagesize = get_arg_val<int32_t>(22);
    // product of window_hw_padded and in_c padded to the tile size (can be multiple tiles)
    const int32_t in_cb_page_nelems_padded = get_arg_val<int32_t>(24);

    // out_w divided by number of out_nelems (== number of blocks per iteration)
    const int32_t out_w_loop_count = get_arg_val<int32_t>(25);
    const uint32_t in_log_base_2_of_page_size = get_arg_val<uint32_t>(26);

    // batch size
    const uint32_t nbatch = get_arg_val<uint32_t>(27);

    const uint32_t in_hw = get_arg_val<uint32_t>(28);

    const uint32_t minus_inf_buffer_addr = get_arg_val<uint32_t>(34);
    const uint32_t minus_inf_buffer_nbytes = get_arg_val<uint32_t>(35);
    const uint32_t in_cb_nrows = get_arg_val<uint32_t>(36);

    // the starting offset for assigned batch input row id (batch_offset)
    uint32_t core_offset_in_row_id = get_arg_val<uint32_t>(37);

    // compile time args
    constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;
    // value of 1 in bf16 in a uin32_t
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(2);
    // number of output elements per iteration == number of blocks per iteration
    constexpr uint32_t out_nelems = get_compile_time_arg_val(3);
    constexpr bool use_pow2 = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id = tt::CB::c_in0; // and tt::CB::c_in1 for split reader
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;

    constexpr uint32_t TILE_HW = 1024;

    // ROW_MAJOR input
    const InterleavedPow2AddrGenFast<is_in_dram> s_in = {
        .bank_base_address = in_addr,
        .log_base_2_of_page_size = in_log_base_2_of_page_size
    };

    // Reduce scalar = 1
    cb_reserve_back(in_scalar_cb_id, 1);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    // fill 1 tile w/ scalar
    fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_HW, bf16_one_u16);
    cb_push_back(in_scalar_cb_id, 1);

    // fill in_cb_id rows with -inf
    uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    const InterleavedPow2AddrGenFast<false> s_const = {     // NOTE: This is always in L1 (hardcoded in host)
        .bank_base_address = minus_inf_buffer_addr,
        .log_base_2_of_page_size = in_log_base_2_of_page_size        // TODO: generalize?, currently hardcorded for 1 row of 32 16b values
    };
    fill_with_val_async(s_const, in_l1_write_addr, in_cb_nrows, in_nbytes_c);
    noc_async_read_barrier();

    // NOTE: batch is folded in

    // DPRINT << "NOC coords 0: " << (uint) my_x[0] << "," << (uint) my_y[0] << ENDL();
    // DPRINT << "NOC coords 1: " << (uint) my_x[1] << "," << (uint) my_y[1] << ENDL();

    uint32_t core_out_w_i_start = get_arg_val<int32_t>(38);
    uint32_t core_out_h_i_start = get_arg_val<int32_t>(39);
    uint32_t nsticks_per_core = get_arg_val<uint32_t>(40);

    uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(42);

    int32_t out_h_i = core_out_h_i_start;
    int32_t out_w_i = core_out_w_i_start;
    int32_t stride_w_multiples = stride_w * out_w_i;
    int32_t stride_h_multiples = stride_h * out_h_i;
    for (uint32_t stick = 0; stick < nsticks_per_core_by_nblocks; ++ stick) {
        cb_reserve_back(in_cb_id, out_nelems);
        uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
        for (uint32_t block = 0; block < out_nelems; ++ block) {
            // for given stick (out_w_i, out_h_i), calculate:
            //      start_h, start_w, end_h, end_w for window on input
            int32_t start_w = stride_w_multiples - pad_w;
            int32_t start_h = stride_h_multiples - pad_h;
            int32_t end_w = start_w + window_w;
            int32_t end_h = start_h + window_h;
            // sanitize the values on edges
            start_w = start_w < 0 ? 0 : start_w;
            start_h = start_h < 0 ? 0 : start_h;
            end_w = end_w > in_w ? in_w : end_w;
            end_h = end_h > in_h ? in_h : end_h;

            // DPRINT << "READ for stick " << stick << " = " << (uint) out_w_i << "," << (uint) out_h_i << " :: " << (uint) start_w << "," << (uint) start_h << "..." << (uint) end_w << "," << (uint) end_h << ENDL();

            // read at most window_hw input rows into CB
            int32_t read_rows = 0;
            uint32_t curr_in_l1_write_addr = in_l1_write_addr;
            uint32_t in_w_multiples = in_w * start_h;
            for (int32_t h = start_h; h < end_h; ++ h, in_w_multiples += in_w) {
                for (int32_t w = start_w; w < end_w; ++ w) {
                    uint32_t in_hw_row_id = core_offset_in_row_id + in_w_multiples + w;
                    // DPRINT << in_hw_row_id << " ";
                    s_in.noc_async_read_page(in_hw_row_id, curr_in_l1_write_addr);
                    curr_in_l1_write_addr += in_nbytes_c;
                    ++ read_rows;
                }
            }
            // DPRINT << ENDL();
            // DPRINT << TileSlice(in_cb_id, 0, srt, true, false);
            // TODO: this should be handled by untilize + edge pad (previous OP)
            if (read_rows < window_hw) {
                // if needed, fill the remainining (window_hw - read_row_id) with -INF
                fill_with_val_async(s_const, curr_in_l1_write_addr, window_hw - read_rows, in_nbytes_c);
            }
            in_l1_write_addr += in_cb_pagesize;

            // increment to next stick
            ++ out_w_i;
            stride_w_multiples += stride_w;
            if (out_w_i == out_w) {
                out_w_i = 0;
                stride_w_multiples = 0;
                ++ out_h_i;
                stride_h_multiples += stride_h;
                if (out_h_i == out_h) {
                    out_h_i = 0;    // new batch starts
                    stride_h_multiples = 0;
                    core_offset_in_row_id += in_hw;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(in_cb_id, out_nelems);
    }
} // kernel_main()

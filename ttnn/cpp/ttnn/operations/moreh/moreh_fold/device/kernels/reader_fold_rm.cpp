// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    int i{0};
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t H = get_arg_val<uint32_t>(3);
    const uint32_t W = get_arg_val<uint32_t>(4);
    const uint32_t kernel_size_h = get_arg_val<uint32_t>(5);
    const uint32_t kernel_size_w = get_arg_val<uint32_t>(6);
    const uint32_t stride_h = get_arg_val<uint32_t>(7);
    const uint32_t stride_w = get_arg_val<uint32_t>(8);
    const uint32_t padding_h = get_arg_val<uint32_t>(9);
    const uint32_t padding_w = get_arg_val<uint32_t>(10);
    const uint32_t dilation_h = get_arg_val<uint32_t>(11);
    const uint32_t dilation_w = get_arg_val<uint32_t>(12);
    const uint32_t LH = get_arg_val<uint32_t>(13);
    const uint32_t LW = get_arg_val<uint32_t>(14);
    const uint32_t input_cb_page_size = get_arg_val<uint32_t>(15);
    const uint32_t output_cb_page_size = get_arg_val<uint32_t>(16);
    const uint32_t start_id = get_arg_val<uint32_t>(17);
    const uint32_t num_units_per_core = get_arg_val<uint32_t>(18);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t onetile = 1;

    uint32_t P = kernel_size_h * kernel_size_w;
    uint32_t l = LH * LW;

    const InterleavedAddrGen<input_is_dram> s0 = {.bank_base_address = input_addr, .page_size = input_cb_page_size};

    for (uint32_t row_id = start_id; row_id < start_id + num_units_per_core; row_id++) {
        cb_reserve_back(output_cb_id, onetile);
        for (uint32_t elem_id = 0; elem_id < W; elem_id++) {
            uint32_t gid = row_id * W + elem_id;
            uint32_t nch = gid / W;
            uint32_t w = gid % W;
            uint32_t nc = nch / H;
            uint32_t h = nch % H;
            uint32_t n = nc / C;
            uint32_t c = nc % C;
            float sum = 0.0f;
            for (uint32_t ph = 0; ph < kernel_size_h; ++ph) {
                for (uint32_t pw = 0; pw < kernel_size_w; ++pw) {
                    uint32_t lhsh = h - ph * dilation_h + padding_h;
                    uint32_t lwsw = w - pw * dilation_w + padding_w;
                    if (lhsh % stride_h != 0) {
                        continue;
                    }
                    if (lwsw % stride_w != 0) {
                        continue;
                    }
                    uint32_t lh = lhsh / stride_h;
                    uint32_t lw = lwsw / stride_w;
                    if (lh < 0 || LH <= lh) {
                        continue;
                    }
                    if (lw < 0 || LW <= lw) {
                        continue;
                    }

                    // Input size is {N, C * kernel_size_h * kernel_size_w, LH * LW} or {C * kernel_size_h *
                    // kernel_size_w, LH * LW}
                    uint32_t input_row_id = n * C * P + (c * P + ph * kernel_size_w + pw);
                    // Read entire row into input_cb
                    cb_reserve_back(input_cb_id, onetile);
                    uint32_t l1_write_addr = get_write_ptr(input_cb_id);
                    uint64_t src_noc_addr = get_noc_addr(input_row_id, s0);
                    noc_async_read(src_noc_addr, l1_write_addr, input_cb_page_size);
                    noc_async_read_barrier();
                    cb_push_back(input_cb_id, onetile);

                    cb_wait_front(input_cb_id, onetile);
#ifdef DTYPE_BFLOAT16
                    uint16_t* input_cb_ptr_uint16 = reinterpret_cast<uint16_t*>(get_read_ptr(input_cb_id));
                    uint16_t bfloat16_value = input_cb_ptr_uint16[lh * LW + lw];
                    uint32_t float_value_as_int = static_cast<uint32_t>(bfloat16_value) << 16;
                    auto tmp = reinterpret_cast<float*>(&float_value_as_int);
                    float value_as_float = *tmp;
                    sum += value_as_float;
#endif
#ifdef DTYPE_FLOAT32
                    auto input_cb_ptr_float = reinterpret_cast<float*>(get_read_ptr(input_cb_id));
                    sum += input_cb_ptr_float[lh * LW + lw];
#endif
                    cb_pop_front(input_cb_id, onetile);
                }
            }
#ifdef DTYPE_BFLOAT16
            uint16_t* output_cb_write_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(output_cb_id));
            auto sum_ptr = reinterpret_cast<uint16_t*>(&sum) + 1;
            output_cb_write_ptr[w] = *sum_ptr;
#endif
#ifdef DTYPE_FLOAT32
            float* output_cb_write_ptr = reinterpret_cast<float*>(get_write_ptr(output_cb_id));
            output_cb_write_ptr[w] = sum;
#endif
        }
        cb_push_back(output_cb_id, onetile);
    }
}

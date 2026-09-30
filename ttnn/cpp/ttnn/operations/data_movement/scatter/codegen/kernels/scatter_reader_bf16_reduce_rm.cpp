// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dedicated ROW_MAJOR BF16 scatter-reduction reader.
//
// Mirrors tt-metal's ScatterReduceBfloat16ProgramFactory: keep one full FP32
// accumulator stick, widen the BF16 input once, apply every duplicate update
// in FP32, then narrow once.  The BRISC-side scatter_writer_rm.cpp continues
// to own input reads and output writes.

#include "scatter_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/numeric/bfloat16.h"
#include <cstdint>

FORCE_INLINE float reduce_fp32(const float current, const uint16_t source_bf16, const uint32_t reduction_mode) {
    const float source = bf16_to_fp32(source_bf16);
    return reduction_mode == 1 ? current + source : current * source;
}

void kernel_main() {
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_stick = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);
    // Args 4..20 are the fixed-width logical page-map descriptor.

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src = get_compile_time_arg_val(3);
    constexpr uint32_t cb_fp32_temp = get_compile_time_arg_val(4);
    constexpr uint32_t input_stick_elems = get_compile_time_arg_val(5);
    constexpr uint32_t index_stick_elems = get_compile_time_arg_val(6);
    constexpr uint32_t index_df_size = get_compile_time_arg_val(7);
    constexpr uint32_t src_df_size = get_compile_time_arg_val(8);
    constexpr uint32_t index_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t src_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t chunk_elems = get_compile_time_arg_val(11);
    constexpr auto index_ta_args = TensorAccessorArgs<12>();
    constexpr auto src_ta_args = TensorAccessorArgs<index_ta_args.next_compile_time_args_offset()>();
    constexpr uint32_t reduction_mode = get_named_compile_time_arg_val("reduction_mode");
    constexpr bool packed_bf16_io = get_named_compile_time_arg_val("packed_bf16_io") != 0;
    static_assert(reduction_mode == 1 || reduction_mode == 2);

    const auto index_accessor = TensorAccessor(index_ta_args, index_addr, index_page_bytes);
    const auto src_accessor = TensorAccessor(src_ta_args, src_addr, src_page_bytes);

    constexpr uint32_t one_page = 1;
    Noc noc;
    CircularBuffer in_cb(cb_input);
    CircularBuffer out_cb(cb_output);
    CircularBuffer index_cb(cb_index);
    CircularBuffer src_cb(cb_src);
    CircularBuffer fp32_cb(cb_fp32_temp);

    index_cb.reserve_back(one_page);
    src_cb.reserve_back(one_page);
    fp32_cb.reserve_back(one_page);
    const uint32_t idx_l1 = index_cb.get_write_ptr();
    const uint32_t src_l1 = src_cb.get_write_ptr();
    volatile tt_l1_ptr float* accum = reinterpret_cast<volatile tt_l1_ptr float*>(fp32_cb.get_write_ptr());

    for (uint32_t s = 0; s < num_sticks; ++s) {
        in_cb.wait_front(one_page);
        out_cb.reserve_back(one_page);
        const volatile tt_l1_ptr uint16_t* input = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_cb.get_read_ptr());
        volatile tt_l1_ptr uint16_t* output = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_cb.get_write_ptr());

        if constexpr (packed_bf16_io) {
            // Every admitted page is 32-byte aligned, hence contains an even
            // number of BF16 values.  Load two lanes per L1 transaction while
            // retaining the same lane order and scalar conversion semantics.
            const volatile tt_l1_ptr uint32_t* input_packed =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(input);
            for (uint32_t i = 0; i < input_stick_elems; i += 2) {
                const uint32_t pair = input_packed[i >> 1];
                accum[i] = bf16_to_fp32(static_cast<uint16_t>(pair));
                accum[i + 1] = bf16_to_fp32(static_cast<uint16_t>(pair >> 16));
            }
        } else {
            for (uint32_t i = 0; i < input_stick_elems; ++i) {
                accum[i] = bf16_to_fp32(input[i]);
            }
        }

        const uint32_t input_stick_id = start_stick + s;
        uint32_t source_stick_id = 0;
        const bool has_src_data = map_scatter_input_page<4>(input_stick_id, source_stick_id);

        for (uint32_t base = 0; has_src_data && base < index_stick_elems; base += chunk_elems) {
            uint32_t this_chunk = chunk_elems;
            if (base + this_chunk > index_stick_elems) {
                this_chunk = index_stick_elems - base;
            }
            noc.async_read(
                index_accessor,
                index_cb,
                this_chunk * index_df_size,
                {.page_id = source_stick_id, .offset_bytes = base * index_df_size},
                {.offset_bytes = 0});
            noc.async_read(
                src_accessor,
                src_cb,
                this_chunk * src_df_size,
                {.page_id = source_stick_id, .offset_bytes = base * src_df_size},
                {.offset_bytes = 0});
            noc.async_read_barrier();

            for (uint32_t i = 0; i < this_chunk; ++i) {
                const uint32_t dest = get_value_from_tile(idx_l1, i, index_df_size);
                if (dest < input_stick_elems) {
                    const uint16_t source = static_cast<uint16_t>(get_value_from_tile(src_l1, i, src_df_size));
                    accum[dest] = reduce_fp32(accum[dest], source, reduction_mode);
                }
            }
        }

        if constexpr (packed_bf16_io) {
            volatile tt_l1_ptr uint32_t* output_packed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output);
            for (uint32_t i = 0; i < input_stick_elems; i += 2) {
                const uint32_t lo = fp32_to_bf16(accum[i]);
                const uint32_t hi = fp32_to_bf16(accum[i + 1]);
                output_packed[i >> 1] = lo | (hi << 16);
            }
        } else {
            for (uint32_t i = 0; i < input_stick_elems; ++i) {
                output[i] = fp32_to_bf16(accum[i]);
            }
        }
        in_cb.pop_front(one_page);
        out_cb.push_back(one_page);
    }
}

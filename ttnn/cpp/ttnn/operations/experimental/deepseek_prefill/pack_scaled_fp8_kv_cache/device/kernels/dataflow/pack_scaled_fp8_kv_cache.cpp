// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t latent_addr = get_arg_val<uint32_t>(0);
    const uint32_t scale_addr = get_arg_val<uint32_t>(1);
    const uint32_t rope_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_row = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(0);
    constexpr uint32_t latent_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t scale_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t rope_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t scale_offset = latent_bytes;
    constexpr uint32_t rope_offset = latent_bytes + scale_bytes;

    constexpr auto latent_args = TensorAccessorArgs<4>();
    constexpr auto scale_args = TensorAccessorArgs<latent_args.next_compile_time_args_offset()>();
    constexpr auto rope_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<rope_args.next_compile_time_args_offset()>();
    const auto latent = TensorAccessor(latent_args, latent_addr);
    const auto scales = TensorAccessor(scale_args, scale_addr);
    const auto rope = TensorAccessor(rope_args, rope_addr);
    const auto output = TensorAccessor(output_args, output_addr);

    Noc noc;
    CircularBuffer scratch(cb_scratch_id);
    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        scratch.reserve_back(1);
        noc.async_read(latent, scratch, latent_bytes, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            latent_bytes,
            {.offset_bytes = 0},
            {.page_id = row});
        noc.async_write_barrier();

        noc.async_read(scales, scratch, scale_bytes, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            scale_bytes,
            {.offset_bytes = 0},
            {.page_id = row, .offset_bytes = scale_offset});
        noc.async_write_barrier();

        noc.async_read(rope, scratch, rope_bytes, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            rope_bytes,
            {.offset_bytes = 0},
            {.page_id = row, .offset_bytes = rope_offset});
        noc.async_write_barrier();
        scratch.push_back(1);
        scratch.pop_front(1);
    }
}

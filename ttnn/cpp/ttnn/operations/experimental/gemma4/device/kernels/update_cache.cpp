// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/experimental/gemma4/request_descriptor.hpp"

void kernel_main() {
    namespace request = ttnn::experimental::gemma4::request;
    constexpr uint32_t index = get_compile_time_arg_val(0);
    constexpr uint32_t span = get_compile_time_arg_val(1);
    constexpr uint32_t input_rows = get_compile_time_arg_val(2);
    constexpr uint32_t width = get_compile_time_arg_val(3);
    constexpr uint32_t cache_rows = get_compile_time_arg_val(4);
    constexpr uint32_t heads = get_compile_time_arg_val(5);
    constexpr uint32_t cache_batches = get_compile_time_arg_val(6);
    constexpr uint32_t layer = get_compile_time_arg_val(7);
    constexpr uint32_t layers = get_compile_time_arg_val(8);
    constexpr uint32_t cp = get_compile_time_arg_val(9);
    constexpr uint32_t rank = get_compile_time_arg_val(10);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(11);
    constexpr auto input_args = TensorAccessorArgs<12>();
    constexpr auto cache_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto descriptor_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0));
    const auto cache = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(1));
    const auto descriptors = TensorAccessor(descriptor_args, get_common_arg_val<uint32_t>(2));
    CircularBuffer scratch(0);
    scratch.reserve_back(1);
    const uint32_t address = scratch.get_write_ptr();
    Noc noc;
    noc.async_read(descriptors, CoreLocalMem<uint8_t>(address), request::bytes, {.page_id = index}, {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const CoreLocalMem<volatile uint32_t> descriptor(address);
    const uint32_t active = descriptor[request::active];
    const uint32_t slot = descriptor[request::cache_slot];
    const uint32_t prefix = descriptor[request::prefix_tokens];
    const uint32_t valid = descriptor[request::valid_tokens];
    const uint32_t offset = descriptor[request::packed_offset_tiles];
    const bool bounds = active == 1 && slot < cache_batches / layers && prefix % 32 == 0 &&
                        prefix < cache_rows * cp * 32 && valid > 0 && valid <= span * cp * 32 &&
                        valid <= cache_rows * cp * 32 - prefix && descriptor[request::packed_span_tiles] == span &&
                        span <= input_rows && offset <= input_rows - span;
    ASSERT(bounds);
    if (!bounds) {
        return;
    }
    const auto local_rows_before = [](uint32_t tile) { return tile / cp + (tile % cp > rank ? 1 : 0); };
    const uint32_t cache_start = local_rows_before(prefix / 32);
    const uint32_t rows = local_rows_before((prefix + valid + 31) / 32) - cache_start;
    const uint32_t batch = slot * layers + layer;
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
        const uint32_t column = tile % width;
        const uint32_t row = tile / width % span;
        const uint32_t head = tile / (width * span);
        if (row >= rows) {
            continue;
        }
        const uint32_t src = (head * input_rows + offset + row) * width + column;
        const uint32_t dst = ((batch * heads + head) * cache_rows + cache_start + row) * width + column;
        noc.async_read(input, CoreLocalMem<uint8_t>(address), page_bytes, {.page_id = src}, {});
        noc.async_read_barrier();
        noc.async_write(CoreLocalMem<uint8_t>(address), cache, page_bytes, {}, {.page_id = dst});
        noc.async_write_barrier();
    }
}

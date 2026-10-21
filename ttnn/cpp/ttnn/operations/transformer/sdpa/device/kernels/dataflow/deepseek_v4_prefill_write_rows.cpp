// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_write_rows: copies this core's input rows to the output rows the row index names.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_write_rows_common.hpp"

namespace dwr = deepseek_v4_prefill_write_rows;

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(dwr::ct_arg::ROW_BYTES);
    constexpr uint32_t num_rows = get_compile_time_arg_val(dwr::ct_arg::NUM_ROWS);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(dwr::ct_arg::CB_ROWS);
    constexpr uint32_t cb_data = get_compile_time_arg_val(dwr::ct_arg::CB_DATA);
    constexpr auto in_args = TensorAccessorArgs<dwr::ct_arg::END, dwr::common_arg::END>();
    constexpr auto out_args =
        TensorAccessorArgs<in_args.next_compile_time_args_offset(), in_args.next_common_runtime_args_offset()>();
    constexpr auto rows_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(dwr::common_arg::INPUT_ADDRESS));
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(dwr::common_arg::OUTPUT_ADDRESS));
    const auto rows = TensorAccessor(rows_args, get_common_arg_val<uint32_t>(dwr::common_arg::ROWS_ADDRESS));
    const uint32_t output_rows = get_common_arg_val<uint32_t>(dwr::common_arg::OUTPUT_ROWS);
    const uint32_t first = get_arg_val<uint32_t>(dwr::core_arg::FIRST_ROW);
    const uint32_t count = get_arg_val<uint32_t>(dwr::core_arg::NUM_ROWS);

    Noc noc;
    experimental::CB rows_cb(cb_rows), data_cb(cb_data);
    const uint32_t rows_l1 = rows_cb.get_write_ptr();
    const uint32_t data_l1 = data_cb.get_write_ptr();
    // The row index is one page; the whole of it is read (the rows are written by the host or an earlier op).
    noc.async_read(rows, CoreLocalMem<uint8_t>(rows_l1), num_rows * sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    invalidate_l1_cache();
    auto* index = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1);

    for (uint32_t done = 0; done < count; done += dwr::BATCH_ROWS) {
        const uint32_t n = count - done < dwr::BATCH_ROWS ? count - done : dwr::BATCH_ROWS;
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_read(
                in, CoreLocalMem<uint8_t>(data_l1 + j * row_bytes), row_bytes, {.page_id = first + done + j}, {});
        }
        noc.async_read_barrier();
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t dst = index[first + done + j];
            if (dst < output_rows) {
                noc.async_write(CoreLocalMem<uint8_t>(data_l1 + j * row_bytes), out, row_bytes, {}, {.page_id = dst});
            }
        }
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Injector x prefetch on NCRISC (NoC0): reads K-blocks [kb_start, kb_end) of x into this core's cb_x_resident
// slot ([m][k] per K-block) and raises the local prefetch semaphore for the writer, which multicasts the slot.
// Injectors are idle cores, so NCRISC has no W stream here. The writer owns cb_x_resident; this kernel only
// writes the slot at the CB base, never pushes.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t prefetch_sem_id = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t kb_start = get_arg_val<uint32_t>(1);
    const uint32_t kb_end = get_arg_val<uint32_t>(2);

    const uint32_t x_page_bytes = get_tile_size(cb_x_resident);
    const auto x = TensorAccessor(x_args, x_addr, x_page_bytes);
    Noc noc;
    CircularBuffer cb_x(cb_x_resident);
    MaybeDeviceZoneScope("ncrisc_x_prefetch");
    uint32_t offset = kb_start * block_m_tiles * k_block_tiles * x_page_bytes;
    for (uint32_t b = kb_start; b < kb_end; ++b) {
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            const uint32_t row_page = m * tensor_k_tiles + b * k_block_tiles;
            for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
                noc.async_read(x, cb_x, x_page_bytes, {.page_id = row_page + kk}, {.offset_bytes = offset});
                offset += x_page_bytes;
            }
        }
    }
    noc.async_read_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(prefetch_sem_id)), 1);
}

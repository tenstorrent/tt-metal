// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tile_padding.hpp"
void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto accessor_args = TensorAccessorArgs<1>();
    auto src = TensorAccessor(accessor_args, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t bytes = get_tile_size(0);
    Noc noc;
    DataflowBuffer cb(0);
    for (uint32_t i = 0; i < count; i += batch) {
        cb.reserve_back(batch);
        for (uint32_t j = 0; j < batch; ++j) {
            noc.async_read(src, cb, bytes, {.page_id = start + i + j}, {.offset_bytes = j * bytes});
        }
        noc.async_read_barrier();
#ifdef SDPA_PREPARE_ROWS
        constexpr uint32_t rows = (SDPA_PREPARE_ROWS + 31) / 32;
        for (uint32_t j = 0; j < batch; ++j) {
            if (((start + i + j) / 4) % rows == rows - 1) {
                zero_tile_padding<2048>(cb.get_write_ptr() + j * bytes, SDPA_PREPARE_ROWS % 32);
            }
        }
#endif
        cb.push_back(batch);
    }
}

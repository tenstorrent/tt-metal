// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp"

// Dataflow-only ROW_MAJOR pad cleanup. A cache page is one token row (including aligned row padding),
// so stream an all-zero L1 row to exactly [valid_global, ceil_pad(valid_global)). This works for BF16
// and FP8_E4M3 because the unpack/compute engine never touches the payload.
void kernel_main() {
    constexpr uint32_t zero_cb = get_compile_time_arg_val(0);
    constexpr uint32_t row_page_bytes = get_compile_time_arg_val(1);
    constexpr auto cache_args = TensorAccessorArgs<2>();

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const ZeroPadRowMajorChipWork w = zero_pad_compute_row_major_chip_work();
    if (w.count == 0) {
        return;
    }

    const auto cache = TensorAccessor(cache_args, cache_addr, row_page_bytes);
    CircularBuffer zero(zero_cb);
    zero.reserve_back(1);

    Noc noc;
    noc.async_write_zeros(zero, row_page_bytes);
    noc.write_zeros_l1_barrier();

    const uint32_t first_page = w.batch_page_base + w.base_local_row;
    const uint32_t last_page = first_page + w.count;
    for (uint32_t page = first_page; page < last_page; ++page) {
        noc.async_write_zeros(cache, row_page_bytes, {.page_id = page}, zero);
    }
    noc.write_zeros_dram_barrier();
}

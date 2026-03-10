// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel (stub)
// Writes untilized RM sticks from c_16 to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"
// Full implementation will also need:
// #include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // Stub: writer kernel
    // Real implementation will:
    // Per tile-row:
    //   1. cb_wait_front(c_16, Wt)
    //   2. Get L1 base via get_read_ptr(c_16)
    //   3. Write 32 sticks to DRAM using TensorAccessor
    //   4. noc_async_write_barrier()
    //   5. cb_pop_front(c_16, Wt)
}

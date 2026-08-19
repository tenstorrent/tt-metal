// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — receiver writer (BRISC).
//
// Phase 7 of the dataflow phase table: drain cb_output_pages (assembled page by
// page by the receiver reader) into this device's output DRAM shard, one page per
// pop, in FIFO order. Depth 2 double-buffers assemble(p+1) against write(p).
//
// Only the RECEIVER device's shard is written; every other coordinate's shard is
// left byte-for-byte as it was on entry (the host seeds the default output with
// ttnn.clone so that statement is total rather than undefined).
//
// Raw NoC (not a helper) by design: ccl_helpers_dataflow.hpp:130-140 lists
// address generation among the things the CCL helper explicitly does NOT own, and
// this side is a plain local L1->DRAM page move with no fabric involvement.
//
// MANDATORY (op_design.md Key Risk #1): the TensorAccessor is built with exactly
// TWO arguments so its per-bank page stride is the compile-time-baked
// buffer.aligned_page_size(). Passing `page_size` as a third argument would set
// the stride to the UNALIGNED logical page size and corrupt every page beyond the
// first bank row whenever page_size % DRAM_ALIGNMENT != 0 (a 96 B or 48 B
// row-major row on Blackhole, whose DRAM alignment is 64 B). page_size is used
// ONLY as the noc_async_write byte count.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    uint32_t ai = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_pages = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);

    // 2-argument ctor: stride == buffer.aligned_page_size() (CT-baked). NOT page_size.
    const auto output = TensorAccessor(output_args, output_addr);

    for (uint32_t p = 0; p < num_pages; ++p) {
        cb_wait_front(cb_output_pages, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_output_pages);
        noc_async_write(l1_read_addr, output.get_noc_addr(p), page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_output_pages, 1);
    }
}

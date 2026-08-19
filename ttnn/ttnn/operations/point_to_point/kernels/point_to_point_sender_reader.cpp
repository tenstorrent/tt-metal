// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — sender reader (NCRISC).
//
// Phase 2 of the dataflow phase table: stream this device's whole input shard
// out of DRAM and into cb_shard_pages, one page per push, in FIFO order. The
// sender writer (BRISC) pops the same pages one at a time and frames them into
// fabric packets, so a depth-2 CB double-buffers read(p+1) against frame(p).
//
// Raw NoC (not a helper) by design: ccl_helpers_dataflow.hpp:130-140 lists
// "address generation (TensorAccessor/ShardedAddrGen is consumed, never
// re-wrapped)" among the things the CCL helper explicitly does NOT own, and this
// side of the op is a plain local DRAM->L1 page move with no fabric involvement.
//
// MANDATORY (op_design.md Key Risk #1): the TensorAccessor is built with exactly
// TWO arguments. Its per-bank page stride must be the compile-time-baked
// buffer.aligned_page_size(); passing `page_size` as a third argument would set
// the stride to the UNALIGNED logical page size and mis-address every page beyond
// the first bank row whenever page_size % DRAM_ALIGNMENT != 0 (e.g. a 96 B or
// 48 B row-major row on Blackhole, whose DRAM alignment is 64 B). page_size is
// used ONLY as the noc_async_read byte count.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_shard_pages = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_pages = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);

    // 2-argument ctor: stride == buffer.aligned_page_size() (CT-baked). NOT page_size.
    const auto input = TensorAccessor(input_args, input_addr);

    for (uint32_t p = 0; p < num_pages; ++p) {
        cb_reserve_back(cb_shard_pages, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_shard_pages);
        noc_async_read(input.get_noc_addr(p), l1_write_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_shard_pages, 1);
    }
}

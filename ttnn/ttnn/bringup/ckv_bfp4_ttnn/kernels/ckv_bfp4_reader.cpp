// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ckv_bfp4 / bfp_round reader for the options prefix and compact (the default path keeps the stock
// reader_unary_interleaved_start_id): this core's tiles in order, one page at a time into cb 0, as the stock reader.
//   CKV_PREFIX       two sources: tile i < split from the prefix, else tile i - split of src ([prefix | src])
//   CKV_COMMON_GX    the core's range from common args (ckv_share.hpp: units of `unit` tiles); else per-core args
// compile-time args: unit, split, [prefix accessor,] src accessor
// args (common with CKV_COMMON_GX, else per core): [prefix addr,] src addr, then per, extra (common) or
// num_pages, start_id (per core)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ckv_share.hpp"

void kernel_main() {
#ifdef CKV_COMMON_GX
#define CKV_ARG get_common_arg_val<uint32_t>
#else
#define CKV_ARG get_arg_val<uint32_t>
#endif
    uint32_t a = 0;
#ifdef CKV_PREFIX
    const uint32_t prefix_addr = CKV_ARG(a++);
#endif
    const uint32_t src_addr = CKV_ARG(a++);
    constexpr uint32_t unit = get_compile_time_arg_val(0);
    constexpr uint32_t split = get_compile_time_arg_val(1);
#ifdef CKV_COMMON_GX
    uint32_t first, count;
    ckv_share(CKV_ARG(a), CKV_ARG(a + 1), CKV_COMMON_GX, first, count);
    const uint32_t num_pages = count * unit, start_id = first * unit;
#else
    const uint32_t num_pages = CKV_ARG(a), start_id = CKV_ARG(a + 1);
#endif
#undef CKV_ARG

    constexpr uint32_t cb_id_in0 = 0;
    const uint32_t page_bytes = get_tile_size(cb_id_in0);
#ifdef CKV_PREFIX
    constexpr auto prefix_args = TensorAccessorArgs<2>();
    constexpr auto src_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    const auto prefix = TensorAccessor(prefix_args, prefix_addr, page_bytes);
#else
    constexpr auto src_args = TensorAccessorArgs<2>();
#endif
    const auto src = TensorAccessor(src_args, src_addr, page_bytes);

    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        cb_reserve_back(cb_id_in0, 1);
        const uint32_t l1 = get_write_ptr(cb_id_in0);
#ifdef CKV_PREFIX
        const uint64_t noc = i < split ? prefix.get_noc_addr(i) : src.get_noc_addr(i - split);
#else
        const uint64_t noc = src.get_noc_addr(i);
#endif
        noc_async_read(noc, l1, page_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}

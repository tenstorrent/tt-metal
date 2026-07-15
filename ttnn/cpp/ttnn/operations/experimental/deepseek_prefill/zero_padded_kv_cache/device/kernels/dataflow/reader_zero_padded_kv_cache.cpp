// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "../zero_padded_kv_cache_common.hpp"

// Reads this chip's boundary seq-tile from the cache into the src CB and builds the bf16 row-mask tile
// (rows [0,row_start) = 1.0, rest = 0.0) into the mask CB. UNCONDITIONALLY pushes src+mask every call
// (even on a chip with no pad work, where it reads the slot's base tile and an all-zero mask): the
// compute kernel always consumes Wt src tiles and the writer always pops the out tiles, so the CB
// protocol is balanced without any reader->compute control handoff. The writer decides what (if
// anything) actually gets written back.
//
// The per-call slot_idx / valid_global reach the kernel one of two ways (HasMeta compile flag):
//   - scalar path: common runtime args 9 / 3 (patched on cache hits).
//   - tensor path: NoC-read on-device, each from its own 1-element uint32 tensor -- slot_idx from the
//     tensor at common arg 10, valid_global (= actual_end) from the tensor at common arg 11 (element 0
//     of each). One TensorAccessorArgs is reused for both (identical layout).
// The body is a template on HasMeta so `if constexpr` discards the metadata branch on the scalar
// program (the metadata TensorAccessorArgs offset is made dependent on HasMeta to stay in range).
template <bool HasMeta>
static void run_reader() {
    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mask_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(2);
    // [3] = has_metadata, [4] = metadata CB index. Cache accessor starts at <5>; the metadata accessor
    // (metadata path only) is appended after it.
    constexpr uint32_t meta_cb = get_compile_time_arg_val(4);
    constexpr auto cache_args = TensorAccessorArgs<5>();

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);

    Noc noc;

    ZeroPadChipWork w;
    if constexpr (HasMeta) {
        // NoC-read slot_idx and valid_global, each element 0 of its own 1-element uint32 tensor:
        // slot_idx tensor address = common arg 10, valid_global tensor address = common arg 11. One
        // accessor (kMetaArgsOffset) is reused for both reads (identical layout).
        constexpr uint32_t kMetaArgsOffset = HasMeta ? cache_args.next_compile_time_args_offset() : 0;
        constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
        const uint32_t slot_idx_addr = get_common_arg_val<uint32_t>(10);
        const uint32_t valid_global_addr = get_common_arg_val<uint32_t>(11);
        CircularBuffer cb_meta(meta_cb);
        cb_meta.reserve_back(1);
        const auto s_slot = TensorAccessor(meta_args, slot_idx_addr);
        noc.async_read(s_slot, cb_meta, 4, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t slot = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
        const auto s_valid = TensorAccessor(meta_args, valid_global_addr);
        noc.async_read(s_valid, cb_meta, 4, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t valid_global = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
        cb_meta.push_back(1);
        w = zero_pad_compute_chip_work(slot, valid_global);
    } else {
        w = zero_pad_compute_chip_work();
    }

    const auto s = TensorAccessor(cache_args, cache_addr, cache_tile_bytes);
    const uint32_t base_page = w.batch_page_base + w.base_local_tile * w.Wt;

    // Read the Wt width-tiles of this chip's boundary seq-tile into the src CB. When this chip has no
    // pad work (count==0 -> base_local_tile==0) this reads the slot's base tile; the writer discards it.
    CircularBuffer src(src_cb);
    src.reserve_back(w.Wt);
    for (uint32_t i = 0; i < w.Wt; ++i) {
        noc.async_read(s, src, cache_tile_bytes, {.page_id = base_page + i}, {.offset_bytes = i * cache_tile_bytes});
    }
    noc.async_read_barrier();
    src.push_back(w.Wt);

    // Build the bf16 row-mask tile in L1 (face layout: 4x 16x16 faces, order TL,TR,BL,BR). When there
    // is no partial (row_start==0) the mask is all-zero, which is harmless: the writer discards it.
    constexpr uint16_t kBf16One = 0x3F80;  // 1.0
    CircularBuffer mask(mask_cb);
    mask.reserve_back(1);
    CoreLocalMem<uint16_t> m(mask.get_write_ptr());
    const uint32_t rs = w.row_start;
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t row_base = (face >= 2) ? 16u : 0u;  // faces 0,1 -> rows 0-15; 2,3 -> rows 16-31
        for (uint32_t fr = 0; fr < 16; ++fr) {
            const uint16_t val = ((row_base + fr) < rs) ? kBf16One : 0u;
            for (uint32_t fc = 0; fc < 16; ++fc) {
                m[face * 256 + fr * 16 + fc] = val;
            }
        }
    }
    mask.push_back(1);
}

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(3);
    run_reader<has_metadata>();
}

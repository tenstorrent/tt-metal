// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2.0 fork of kv_cache/.../reader_fill_cache_interleaved_start_id.cpp, so deepseek_prefill can move to
// D2.0 without dragging the (still Device 1.x) kv_cache op along.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

// `kv_actual_global` reaches this kernel the same two ways the writer's does, selected by the
// `has_metadata` compile-time flag: read on-device from the 1-element uint32 tensor whose raw address is
// common arg 7 (metadata/traceable path), or straight out of common arg 7 as a value (scalar path). The
// reader NEEDS the real value even under tp_axis -- its source-row mapping is derived from the chunk
// start -- which is why the metadata path had to grow this read before tp_axis could be allowed on it.
//
// Templated on HasMeta so `if constexpr` DISCARDS the unused branch: kernel_main is not a template, so
// an `if constexpr` there would still instantiate the metadata TensorAccessor and fail to compile the
// scalar program, which carries no metadata accessor args.
template <bool HasMeta>
static void run_reader() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);  // per-core; buffers arrive as Buffer* -> addresses
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(2);

    const uint32_t linear_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t linear_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);  // stripe: page-rows THIS chip writes
    const uint32_t input_Ht = get_common_arg_val<uint32_t>(3);       // window: page-rows the input holds
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(4);
    const uint32_t tp_factor = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt = get_common_arg_val<uint32_t>(6);
    // Common arg 7 is the value on the scalar path and the 1-element tensor's ADDRESS on the metadata
    // path; resolved below, next to the mapping that consumes it.

    constexpr uint32_t tile_height = get_compile_time_arg_val(0);
    // [1] = has_metadata (consumed as the HasMeta template param), [2] = metadata scratch CB (0 on the
    // scalar path). The source accessor therefore starts at 3, and on the metadata path ONE metadata
    // accessor follows it.
    constexpr auto src_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_id_in0 = 0;
    CircularBuffer cb_in0(cb_id_in0);

#ifdef INPUT_SHARDED
    cb_in0.reserve_back(num_pages);
    cb_in0.push_back(num_pages);
#else
    Noc noc;
    uint32_t kv_actual_global;
    if constexpr (HasMeta) {
        // Same on-device read the writer performs, so both derive the chunk start from ONE value and
        // cannot disagree on a non-window-aligned start. Own scratch CB: the writer is doing its own
        // metadata reads into kMetaCbIndex concurrently on this core.
        constexpr uint32_t cb_id_meta = get_compile_time_arg_val(2);
        constexpr uint32_t kMetadataReadBytes = 4;
        // Gate the offset on HasMeta so this is a *dependent* template-id -- the scalar program has no
        // metadata accessor and must not name a fixed out-of-range compile-arg offset here.
        constexpr uint32_t kMetaArgsOffset = HasMeta ? src_args.next_compile_time_args_offset() : 0;
        constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
        CircularBuffer cb_meta(cb_id_meta);
        cb_meta.reserve_back(1);
        const auto s_kv = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(7));
        noc.async_read(s_kv, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // Fixed DRAM address reused every chunk, so the RISC data cache can hold the previous chunk's
        // value for this L1 line; the barrier orders the DMA but volatile still reads cache.
        invalidate_l1_cache();
        kv_actual_global = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
        cb_meta.push_back(1);
    } else {
        kv_actual_global = get_common_arg_val<uint32_t>(7);  // per-call; patched on cache hits
    }

    // Source-row mapping. The TP-replicated input holds one SP chip's whole window (input_Ht rows, in the
    // writer's rotated order) and this chip owns one chunk_local_t stripe of it. Inverting that rotation
    // gives src(j) = base + j, plus `jump` once j leaves the stripe -- two runs only on the start's chip.
    const uint32_t start_t = kv_actual_global / tile_height;  // chunk start, in page-rows
    // Chunk start within this chip's stripe / its SP group's window; nonzero only on the chip that holds it
    // (the stripe test is the writer's own expression, so this matches its update_idxt).
    const uint32_t start_in_stripe =
        (linear_coord == (start_t / chunk_local_t) % linear_factor) ? start_t % chunk_local_t : 0;
    const uint32_t start_in_window =
        (linear_coord / tp_factor == (start_t / input_Ht) % sp_factor) ? start_t % input_Ht : 0;
    // + input_Ht keeps the numerator positive (start_in_window < input_Ht); sum < 2*input_Ht, one mod.
    const uint32_t base =
        ((linear_coord % tp_factor) * chunk_local_t + start_in_stripe + input_Ht - start_in_window) % input_Ht;
    const uint32_t jump = chunk_local_t * (tp_factor - 1);  // skip the other TP stripes; 0 at tp_factor == 1

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(src_args, src_addr);
    // CB page size, NOT get_tile_size(): in ROW_MAJOR a page is one token row, and a tile-size read
    // overruns the CB into the writer's metadata scratch. The writer derives its page bytes the same way.
    const uint32_t src_page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // Destination order (block j, then its Wt pages) -- the writer consumes the CB in lockstep.
    const uint32_t n_blocks = num_pages / Wt;
    for (uint32_t k = 0; k < n_blocks; ++k) {
        const uint32_t j = core_blocks_written + k;
        const uint32_t src_page0 = (base + j + (j + start_in_stripe >= chunk_local_t ? jump : 0)) * Wt;
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_in0.reserve_back(onetile);
            noc.async_read(s, cb_in0, src_page_bytes, {.page_id = src_page0 + w}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
        }
    }
#endif
}

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(1) != 0;
    if constexpr (has_metadata) {
        run_reader<true>();
    } else {
        run_reader<false>();
    }
}

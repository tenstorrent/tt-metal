// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2.0 fork of ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp
// for the deepseek_prefill update_padded_kv_cache op. The kv_cache version is still on Device 1.x; this fork lets
// deepseek_prefill flip cleanly to D2.0 without dragging the entire kv_cache op along.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    // Per-core runtime args (buffers arrive as Buffer* bindings -> addresses).
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(2);

    // Common runtime args: structural per cached program, except kv_actual_global (index 7), which is
    // the per-call value patched on cache hits by override_runtime_arguments -- same contract as the
    // writer. The reader needs it because WHICH input rows this chip owns depends on the chunk start
    // (see the source-row mapping below), not just on its mesh position.
    const uint32_t linear_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t linear_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);  // page-rows THIS chip writes (fine stripe)
    const uint32_t input_Ht = get_common_arg_val<uint32_t>(3);       // input page-rows (SP chip's coarse window)
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(4);
    const uint32_t tp_factor = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt = get_common_arg_val<uint32_t>(6);
    const uint32_t kv_actual_global = get_common_arg_val<uint32_t>(7);

    constexpr uint32_t tile_height = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_in0 = 0;
    CircularBuffer cb_in0(cb_id_in0);

#ifdef INPUT_SHARDED
    cb_in0.reserve_back(num_pages);
    cb_in0.push_back(num_pages);
#else
    // ---- Source-row mapping ---------------------------------------------------------------------
    // The input is TP-REPLICATED: it holds SP chip s's whole coarse window of input_Ht page-rows, in
    // the writer's ROTATED order. This chip (linear rank L = s*tp + t) persists only the
    // chunk_local_t rows whose GLOBAL position belongs to L. A plain contiguous t*chunk_local_t slice
    // of that window is correct only when the coarse start is stripe-aligned; with a sub-stripe start
    // offset the fine (stripe-granularity) ownership boundaries fall INSIDE the coarse slice, so the
    // assignment shears and each chip persists rows belonging to its neighbour.
    //
    // Exact mapping. Coarse-local row lr of SP chip s holds global
    // (lr/input_Ht)*chunk_global + s*input_Ht + (lr%input_Ht), so with off = lr % input_Ht that global
    // lands on linear chip s*tp + off/chunk_local_t at local row (lr/input_Ht)*chunk_local_t +
    // off%chunk_local_t. Inverting over this chip's chunk_local_t destination rows (which the writer
    // emits as CONTIGUOUS local rows from its update_idxt) gives, for destination row j:
    //     src_block(j) = base + j          for j <  first
    //                    base + j + jump   for j >= first
    // i.e. one contiguous run, or two when this chip straddles the coarse slab boundary. `u0` is the
    // coarse start offset (nonzero only for the SP chip owning the boundary) and `o` is the fine one
    // (nonzero only for the single linear chip owning it), and o == update_idxt % chunk_local_t, which
    // is why `base` collapses to 0 exactly for that chip.
    const uint32_t kv_t = kv_actual_global / tile_height;
    // Fine-granularity boundary chip -- identical expression to the writer's, so `o` matches its update_idxt.
    const uint32_t boundary_chip_fine = (kv_t / chunk_local_t) % linear_factor;
    const uint32_t o = (linear_coord == boundary_chip_fine) ? (kv_t % chunk_local_t) : 0;
    // Coarse-granularity start offset of this chip's SP peer group.
    const uint32_t sp_coord = linear_coord / tp_factor;
    const uint32_t tp_coord = linear_coord - sp_coord * tp_factor;
    const uint32_t boundary_chip_coarse = (kv_t / input_Ht) % sp_factor;
    const uint32_t u0 = (sp_coord == boundary_chip_coarse) ? (kv_t % input_Ht) : 0;
    // + input_Ht keeps the numerator positive (u0 < input_Ht); the sum stays below 2*input_Ht, so one mod.
    const uint32_t base = (tp_coord * chunk_local_t + o + input_Ht - u0) % input_Ht;
    const uint32_t first = chunk_local_t - o;
    const uint32_t jump = input_Ht - chunk_local_t;  // 0 when tp_factor == 1 -> single run, as before

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(src_args, src_addr);
    Noc noc;
    // Page bytes must come from the CB's configured page size, NOT get_tile_size(): the op runs in
    // ROW_MAJOR too, where a page is one token row (head_dim * element bytes) while get_tile_size()
    // still reports the dtype's 32x32 tile size. Reading tile-size bytes into a row-major page slot
    // over-reads the source page and overruns the CB in L1 (it scribbled the writer's metadata
    // scratch, which sits directly after this CB, corrupting the on-device slot_idx /
    // kv_actual_global read). The writer derives its page bytes the same way.
    const uint32_t src_page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // Push in destination order (block j, then its Wt width pages) -- the writer consumes the CB in
    // exactly that order, so the two stay in lockstep.
    const uint32_t n_blocks = num_pages / Wt;
    for (uint32_t k = 0; k < n_blocks; ++k) {
        const uint32_t j = core_blocks_written + k;
        const uint32_t src_page0 = (base + j + (j >= first ? jump : 0)) * Wt;
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_in0.reserve_back(onetile);
            noc.async_read(s, cb_in0, src_page_bytes, {.page_id = src_page0 + w}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
        }
    }
#endif
}

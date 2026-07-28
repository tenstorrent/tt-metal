// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1) — op_design.md §4.1 and §4.2.
//
// Mirrors the reader's row-block/chunk nest exactly: per core, per row-block of
// HT_BLOCK tile-rows, drain NW chunks of WT_CHUNK W-tiles of this core's W
// slice. The TILE path writes whole tile pages with one barrier per chunk (the
// batched twin of the reader's batched chunk read); the ROW_MAJOR path goes
// through dataflow_kernel_lib::write_sticks_after_untilize, which skips the L1
// W-padding and writes only the valid sticks of a short last tile-row — that is
// what makes non-aligned H/W native with no host-side slice.
//
// SHARDED output (SHARDED_OUT): there is NO write at all. cb_output_tiles is
// placed directly on the core's own L1 output shard, so compute's pack lands in
// the output tensor in place — zero-copy, no NoC traffic.
//
// CROSS-CORE W-SPLIT (W_SPLIT): this kernel owns the *gather* half of the
// combine. Per row-block it ships this core's raw sum(x^2) tiles into slot
// `slot` of the group root's cb_group_partials and bumps the root's gather
// counter; the root's reader waits for all CW of them (see rms_norm_reader.cpp).
// Many-to-one has no multicast form and no helper covers it, so the leg is a
// plain noc_async_write + barrier + noc_semaphore_inc
// (references/cross_core_reduction_design.md §1/§7).
//
// Helper substitution note: the TILE path uses TensorAccessor +
// noc_async_write_page directly because write_sticks_after_untilize writes
// *sticks* produced by the untilize helper, and the TILE path never untilizes —
// its input contract cannot be met. op_design.md §7 mandates exactly this.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
}  // namespace

namespace dkl = dataflow_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    // args 1..5 are the reader/compute regime flags; kept in one shared layout.
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // arg 11 (X_READ_CHUNKS) is a read-side knob; the writer's granularity is
    // set by the compute output CB, which always publishes one chunk at a time.
    // ---- geometry ----
    constexpr uint32_t CHUNK_ROW_BYTES = get_compile_time_arg_val(13);
    constexpr uint32_t LAST_ROW_BYTES = get_compile_time_arg_val(14);
    constexpr uint32_t TOTAL_STICKS = get_compile_time_arg_val(17);
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(19);
    constexpr uint32_t WT_STRIDE = get_compile_time_arg_val(20);
    constexpr bool SHARDED_OUT = get_compile_time_arg_val(22) != 0;
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(23);

    // 24..33 are the two multicast-family CT blocks (reader-side; the writer
    // shares the layout so a knob cannot drift between them).
    constexpr auto out_args = TensorAccessorArgs<34>();

    static_assert(WT_LAST == WT_CHUNK, "writer assumes uniform chunk widths");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t wt_start = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t root_x = get_arg_val<uint32_t>(5);
    const uint32_t root_y = get_arg_val<uint32_t>(6);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(7);

    // Filler core (inside the multicast rectangle, owns no data).
    if (num_tile_rows == 0) {
        return;
    }

    const auto out_acc = TensorAccessor(out_args, dst_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    const uint32_t chunk0 = wt_start / WT_CHUNK;

    // Gather target: the root's cb_group_partials. Every core declares that CB
    // at the same size, so it sits at the same L1 offset everywhere and this
    // core's own write pointer IS the root's base address.
    [[maybe_unused]] const uint32_t partial_tile_bytes = get_tile_size(cb_partial_out);
    [[maybe_unused]] const uint32_t gather_base = get_write_ptr(cb_group_partials);
    [[maybe_unused]] const uint32_t gather_sem_addr = get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER);

    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        const uint32_t row0 = start_tile_row + hb * HT_BLOCK;
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }

        uint32_t valid_rows = ht * 32u;
        if constexpr (IS_RM) {
            const uint32_t remaining = TOTAL_STICKS - row0 * 32u;
            if (remaining < valid_rows) {
                valid_rows = remaining;
            }
        }

        // ---- gather leg: ship this core's raw partial sums to the root -----
        // Laid out h-major (tile h*CW + slot) so the root's combine reads them
        // as a contiguous (ht x CW) block.
        if constexpr (W_SPLIT) {
            cb_wait_front(cb_partial_out, ht);
            uint32_t src = get_read_ptr(cb_partial_out);
            for (uint32_t h = 0; h < ht; ++h) {
                const uint64_t dst = get_noc_addr(root_x, root_y, gather_base + (h * CW + slot) * partial_tile_bytes);
                noc_async_write(src, dst, partial_tile_bytes);
                src += partial_tile_bytes;
            }
            // The data must have LANDED before the root sees the counter move.
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(root_x, root_y, gather_sem_addr), 1);
            cb_pop_front(cb_partial_out, ht);
        }

        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (SHARDED_OUT) {
                // Zero-copy: compute packed straight into the output shard.
                continue;
            } else if constexpr (IS_RM) {
                const uint32_t rb = (is_last_w_core && wc + 1 == NW) ? LAST_ROW_BYTES : CHUNK_ROW_BYTES;
                dkl::write_sticks_after_untilize<cb_output_rm>(
                    out_acc, valid_rows, rb, row0 * 32u, (chunk0 + wc) * CHUNK_ROW_BYTES);
            } else {
                const uint32_t n = ht * WT_CHUNK;
                cb_wait_front(cb_output_tiles, n);
                uint32_t addr = get_read_ptr(cb_output_tiles);
                for (uint32_t h = 0; h < ht; ++h) {
                    const uint32_t base_tile = (row0 + h) * WT_STRIDE + wt_start + wc * WT_CHUNK;
                    for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                        noc_async_write_page(base_tile + t, out_acc, addr);
                        addr += out_tile_bytes;
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, n);
            }
        }
    }

    // noc_semaphore_inc issues a NON-POSTED atomic; leaving one outstanding at
    // kernel exit leaves the core's NoC transaction counters unbalanced, which
    // the dispatcher never sees drain (measured as a device hang on the sharded
    // cells, whose writer has no trailing write barrier to absorb it).
    if constexpr (W_SPLIT) {
        noc_async_atomic_barrier();
    }
}

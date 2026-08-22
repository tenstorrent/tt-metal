// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

/*
Mergesort row engine writer: drains one K=2048-element page per run for values
(bf16) and indices (u32) and writes them to the output row.

pack_untilize emits a K-run as 16-element slices in face-pair order
([top-left, bottom-left, top-right, bottom-right] per tile column — the same
order topk_large_indices' writer handles), so the writer permutes slices on
the way out:
  ROW_MAJOR: permute into a scratch page, then one contiguous row write.
  TILE:      each output slice IS one face row of an output tile, so the
             permutation folds into the scatter — one 32 B (values) / 64 B
             (indices) write per slice, no scratch pass.

NARROW_INDICES (unstable-contract cells, issue #33492 roadmap item 3): the
engine's index words are natively u32 (fused tags live in 32-bit DEST and the
u16-in-32-bit-DEST combination has no working pack path), but the public
unstable index dtype at these widths is UINT16 — so this RISC narrows each
u32 word to u16 into the index scratch page before the output write:
  TILE:      straight narrowing (slice order preserved), then the same
             face-row scatter with 32 B slices at u16-tile offsets.
  ROW_MAJOR: the slice permute is fused into the narrowing loop (the NoC
             permute pass is skipped), then one contiguous row write.
*/

namespace {

constexpr uint32_t K = 2048;
constexpr uint32_t SLICE_ELEMENTS = 16;
constexpr uint32_t SLICES_PER_RUN = K / SLICE_ELEMENTS;  // 128

// pack_untilize face-pair slice order -> linear slice order.
FORCE_INLINE uint32_t source_slice(uint32_t dst_slice) {
    const uint32_t tile_col = dst_slice >> 2;
    const uint32_t face_col = dst_slice & 0x1;
    const uint32_t face_row_offset = (dst_slice & 0x2) ? SLICES_PER_RUN / 2 : 0;
    return (2 * tile_col) + face_col + face_row_offset;
}

#ifdef IS_ROW_MAJOR
// Permute one K-run page into linear order inside a scratch page via local
// NoC reads (same mechanism as topk_large_indices' writer).
template <uint32_t slice_bytes>
FORCE_INLINE void permute_run_to_scratch(DataflowBuffer& src_cb, DataflowBuffer& scratch_cb, const Noc& noc) {
    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, slice_bytes, local_src(src_base));
    for (uint32_t dst_slice = 0; dst_slice < SLICES_PER_RUN; ++dst_slice) {
        const uint32_t src_addr = src_base + source_slice(dst_slice) * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{}, CoreLocalMem<uint32_t>(dst_addr), slice_bytes, local_src(src_addr), {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t start_row = get_arg(args::start_row);
    const uint32_t num_rows = get_arg(args::num_rows);

    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t num_chunks = get_arg(args::num_chunks);

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t value_slice_bytes = SLICE_ELEMENTS * 2;  // bf16
    constexpr uint32_t index_slice_bytes = SLICE_ELEMENTS * 4;  // u32

    const auto value_accessor = TensorAccessor(tensor::value_tensor);
    const auto index_accessor = TensorAccessor(tensor::index_tensor);
    DataflowBuffer values_out_dfb(dfb::values_out);
    DataflowBuffer indices_out_dfb(dfb::indices_out);
#ifdef IS_ROW_MAJOR
    DataflowBuffer values_scratch_dfb(dfb::values_scratch);
#endif
#if defined(IS_ROW_MAJOR) || defined(NARROW_INDICES)
    DataflowBuffer indices_scratch_dfb(dfb::indices_scratch);
#endif
    Noc noc;

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
#ifndef IS_ROW_MAJOR
        const uint32_t tile_row = row / TILE_H;
        const uint32_t r = row % TILE_H;
        // Byte offsets of in-tile row r's two face rows in a bf16 / u32 tile.
        const uint32_t value_fo = ((r >= 16) ? 1024u : 0u) + (r % 16) * 32u;
#ifdef NARROW_INDICES
        // u16 index tiles share the bf16 tile geometry (2 B elements).
        const uint32_t index_fo = value_fo;
        constexpr uint32_t index_face_pitch = 512u;
#else
        const uint32_t index_fo = ((r >= 16) ? 2048u : 0u) + (r % 16) * 64u;
        constexpr uint32_t index_face_pitch = 1024u;
#endif
        constexpr uint32_t tiles_per_run = K / TILE_H;  // 64
#endif

        // ---- Values: one page per run, runs are already in output order ----
        for (uint32_t run = 0; run < num_chunks; ++run) {
            values_out_dfb.wait_front(1);
#ifdef IS_ROW_MAJOR
            values_scratch_dfb.reserve_back(1);
            permute_run_to_scratch<value_slice_bytes>(values_out_dfb, values_scratch_dfb, noc);
            values_scratch_dfb.push_back(1);
            values_scratch_dfb.wait_front(1);
            noc.async_write(
                values_scratch_dfb,
                value_accessor,
                K * 2,
                {.offset_bytes = 0},
                {.page_id = row, .offset_bytes = run * K * 2});
            noc.async_writes_flushed();
            values_scratch_dfb.pop_front(1);
#else
            const uint32_t src_base_offset = 0;
            const uint32_t tile_base = tile_row * Wt + run * tiles_per_run;
            for (uint32_t t = 0; t < tiles_per_run; ++t) {
                // Output tile t's two face rows = output slices 2t and 2t+1.
                noc.async_write(
                    values_out_dfb,
                    value_accessor,
                    value_slice_bytes,
                    {.offset_bytes = src_base_offset + source_slice(2 * t) * value_slice_bytes},
                    {.page_id = tile_base + t, .offset_bytes = value_fo});
                noc.async_write(
                    values_out_dfb,
                    value_accessor,
                    value_slice_bytes,
                    {.offset_bytes = src_base_offset + source_slice(2 * t + 1) * value_slice_bytes},
                    {.page_id = tile_base + t, .offset_bytes = value_fo + 512});
            }
            noc.async_writes_flushed();
#endif
            values_out_dfb.pop_front(1);
        }

        // ---- Indices: same structure (u32 slices, or u16 after narrowing) ----
        for (uint32_t run = 0; run < num_chunks; ++run) {
            indices_out_dfb.wait_front(1);
#ifdef NARROW_INDICES
            // Narrow the run's u32 index words to u16 into the index scratch page.
            // Non-volatile pointers on purpose: the source page is stable (pushed by the
            // packer and wait_front'ed above), the destination is private until the
            // __sync_synchronize below, and these RISC-V cores have no data cache — so
            // letting the compiler batch/unroll the L1 accesses is safe and saves
            // several cycles per element on this hot loop.
            indices_scratch_dfb.reserve_back(1);
            {
                const uint32_t* __restrict src = reinterpret_cast<const uint32_t*>(indices_out_dfb.get_read_ptr());
                uint32_t* __restrict dst = reinterpret_cast<uint32_t*>(indices_scratch_dfb.get_write_ptr());
#ifdef IS_ROW_MAJOR
                // Fuse the face-pair -> linear slice permute into the narrowing.
                for (uint32_t dst_slice = 0; dst_slice < SLICES_PER_RUN; ++dst_slice) {
                    const uint32_t* __restrict s = src + source_slice(dst_slice) * SLICE_ELEMENTS;
                    uint32_t* __restrict d = dst + dst_slice * (SLICE_ELEMENTS / 2);
#pragma GCC unroll 8
                    for (uint32_t j = 0; j < SLICE_ELEMENTS / 2; ++j) {
                        d[j] = (s[2 * j] & 0xFFFFu) | (s[2 * j + 1] << 16);
                    }
                }
#else
                // Straight narrowing; the slice order is preserved for the scatter below.
#pragma GCC unroll 8
                for (uint32_t i = 0; i < K / 2; ++i) {
                    dst[i] = (src[2 * i] & 0xFFFFu) | (src[2 * i + 1] << 16);
                }
#endif
            }
            // Drain the RISC-V store queue before the NoC reads the scratch page
            // (RISC-V stores and the NoC are independent L1 clients).
            __sync_synchronize();
            indices_scratch_dfb.push_back(1);
            indices_scratch_dfb.wait_front(1);
#ifdef IS_ROW_MAJOR
            noc.async_write(
                indices_scratch_dfb,
                index_accessor,
                K * 2,
                {.offset_bytes = 0},
                {.page_id = row, .offset_bytes = run * K * 2});
#else
            constexpr uint32_t out_slice_bytes = SLICE_ELEMENTS * 2;  // u16
            const uint32_t tile_base = tile_row * Wt + run * tiles_per_run;
            for (uint32_t t = 0; t < tiles_per_run; ++t) {
                noc.async_write(
                    indices_scratch_dfb,
                    index_accessor,
                    out_slice_bytes,
                    {.offset_bytes = source_slice(2 * t) * out_slice_bytes},
                    {.page_id = tile_base + t, .offset_bytes = index_fo});
                noc.async_write(
                    indices_scratch_dfb,
                    index_accessor,
                    out_slice_bytes,
                    {.offset_bytes = source_slice(2 * t + 1) * out_slice_bytes},
                    {.page_id = tile_base + t, .offset_bytes = index_fo + index_face_pitch});
            }
#endif
            noc.async_writes_flushed();
            indices_scratch_dfb.pop_front(1);
#else  // !NARROW_INDICES
#ifdef IS_ROW_MAJOR
            indices_scratch_dfb.reserve_back(1);
            permute_run_to_scratch<index_slice_bytes>(indices_out_dfb, indices_scratch_dfb, noc);
            indices_scratch_dfb.push_back(1);
            indices_scratch_dfb.wait_front(1);
            noc.async_write(
                indices_scratch_dfb,
                index_accessor,
                K * 4,
                {.offset_bytes = 0},
                {.page_id = row, .offset_bytes = run * K * 4});
            noc.async_writes_flushed();
            indices_scratch_dfb.pop_front(1);
#else
            const uint32_t tile_base = tile_row * Wt + run * tiles_per_run;
            for (uint32_t t = 0; t < tiles_per_run; ++t) {
                noc.async_write(
                    indices_out_dfb,
                    index_accessor,
                    index_slice_bytes,
                    {.offset_bytes = source_slice(2 * t) * index_slice_bytes},
                    {.page_id = tile_base + t, .offset_bytes = index_fo});
                noc.async_write(
                    indices_out_dfb,
                    index_accessor,
                    index_slice_bytes,
                    {.offset_bytes = source_slice(2 * t + 1) * index_slice_bytes},
                    {.page_id = tile_base + t, .offset_bytes = index_fo + index_face_pitch});
            }
            noc.async_writes_flushed();
#endif
#endif  // NARROW_INDICES
            indices_out_dfb.pop_front(1);
        }
    }

    noc.async_write_barrier();
}

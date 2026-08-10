// Shard-aware destination write for ROW_MAJOR width/block-sharded outputs.
//
// WHY THIS FILE EXISTS
// --------------------
// Our writers address a destination as `{.page_id = <logical row index>}`. That
// identity holds only while a destination page IS a row. tt-metal sets the page
// shape in `RowMajorPageConfig::get_page_shape`
// (tt_metal/impl/tensor/spec/layout/page_config.cpp:190-214) to
// `Shape2D(1, physical_shard_size.width())` whenever a ROW_MAJOR tensor is
// WIDTH- or BLOCK-sharded -- the page is the SHARD width. A logical row then
// spans `ceil(W / shard_w)` pages, `page_id = row` names the wrong bytes, and
// the plain write SCATTERS the tensor.
//
// tt-metal ships a helper for this
// (`tt::data_movement::common::noc_async_write_sharded`,
// ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:264-297).
// WE DELIBERATELY DO NOT CALL IT. It has three defects, each documented and
// verified from source in docs/TTMETAL_NOC_WRITE_SHARDED_BUG.md, and each is
// structurally avoided below:
//
//   (A) It strides the source and counts pages by `get_aligned_page_size()`,
//       which is `align(page_size, buffer_alignment)` -- the BANK STRIDE, not
//       the payload. A page slot is `aligned_page_size` bytes of which only the
//       first `page_size` are real data (tensor_accessor.h:310). Whenever the
//       two differ the loop undercounts pages and silently leaves the tail of
//       every row stale.
//       => We stride by LOGICAL_PAGE_SIZE, supplied by the host from
//          `buffer->page_size()`, never by the accessor's aligned size.
//
//   (B) `write_size = min(size - i * page_size, page_size - sharded_offset)`
//       ignores the initial `sharded_offset`, so the first chunk is short by
//       `offset` and the left term WRAPS on uint32 once `i*page_size > size`
//       (page_size=64, offset=48, size=40 -> 0xFFFFFFE8 -> a full-page write
//       that over-reads the L1 source).
//       => We track `remaining` explicitly in a `while (remaining)` loop. The
//          first chunk is `LOGICAL_PAGE_SIZE - off`, every later chunk starts at
//          off == 0, and `remaining -= chunk` cannot underflow because
//          `chunk <= remaining` by construction.
//
//   (C) It derives `pages_per_row` from `dspec.tensor_shape()[rank-1]`. A
//       one-page-tall shard makes the host-side rank squeeze collapse the dspec
//       to rank 1 (buffer_distribution_spec.cpp:330-380), `pages_per_row` reads
//       1, and the helper falls through to a single unsplit write at a row index
//       -- wrong page AND a cross-page overrun. The information is destroyed
//       host-side; the helper cannot recover it.
//       => PAGES_PER_ROW is a HOST-supplied compile-time constant. We never read
//          the dspec, so the squeeze cannot mislead us.
//
// KILL SWITCH / ZERO-DIFF WHEN OFF
// --------------------------------
// PAGES_PER_ROW <= 1 means "a page IS a row". Because both parameters are
// compile-time constants, the `if constexpr` below makes the disabled path emit
// EXACTLY the single `noc.async_write` the caller emitted before this header
// existed -- same instruction, same operands, no loop, no code-size change. The
// host passes (0, 0) whenever the shard-split route is not taken, so a writer
// that includes this header is byte-identical to its pre-change self on every
// route we have not deliberately enabled.

#pragma once

#include <stdint.h>

namespace ttdm {

// Write `size` bytes to logical destination row `row`, starting at byte
// `dst_offset` within that row, sourced from `src` at byte `src_offset`.
//
// `src` is whatever source descriptor the caller already passes to
// `noc.async_write` (a CircularBuffer, a CoreLocalMem<uint32_t>, ...); `acc` is
// the destination TensorAccessor. Both are forwarded untouched, so this header
// imposes no new ABI on the caller beyond the two compile-time constants.
//
// PRECONDITIONS when PAGES_PER_ROW > 1 -- all enforced host-side by
// `common.codegen_common.rm_shard_pages.shard_split_write_is_exact`:
//   * LOGICAL_PAGE_SIZE == the destination buffer's `page_size()`, and
//     `align(page_size, buffer_alignment) == page_size` (no inter-page padding).
//     For an L1 destination that is exactly `P % 16 == 0`, which is also
//     NOC_L1_WRITE_ALIGNMENT_BYTES -- so every `src_off` this loop produces
//     stays 16-byte aligned, as the NoC requires.
//   * PAGES_PER_ROW == ceil(row_width_elements / shard_width_elements), taken
//     from the page definition and NOT from the (possibly squeezed) dspec.
//   * dst_offset + size <= PAGES_PER_ROW * LOGICAL_PAGE_SIZE (the write stays
//     inside one logical row).
template <
    uint32_t PAGES_PER_ROW,
    uint32_t LOGICAL_PAGE_SIZE,
    typename NocT,
    typename SrcT,
    typename AccT>
FORCE_INLINE void noc_write_row_split(
    NocT& noc,
    const SrcT& src,
    uint32_t src_offset,
    const AccT& acc,
    uint32_t row,
    uint32_t dst_offset,
    uint32_t size) {
    if constexpr (PAGES_PER_ROW <= 1) {
        // A page is a row: the historical single write, unchanged.
        noc.async_write(
            src, acc, size, {.offset_bytes = src_offset}, {.page_id = row, .offset_bytes = dst_offset});
    } else {
        // A logical row occupies PAGES_PER_ROW consecutive page indices. Page k
        // of the row holds row bytes [k*L, (k+1)*L) where L is the LOGICAL page
        // size. The accessor turns {page_id, offset} into
        //   bank_start + base + bank_page_offset * aligned_page_size + offset
        // (tensor_accessor.h:310), so addressing stays correct for any
        // aligned_page_size as long as offset < L <= aligned_page_size.
        uint32_t page = row * PAGES_PER_ROW + dst_offset / LOGICAL_PAGE_SIZE;
        uint32_t off = dst_offset % LOGICAL_PAGE_SIZE;
        uint32_t src_off = src_offset;
        uint32_t remaining = size;
        while (remaining > 0) {
            // First chunk is short by `off`; every later chunk starts at 0.
            // Never `size - i * L`: that both understates the tail and wraps.
            uint32_t chunk = LOGICAL_PAGE_SIZE - off;
            if (chunk > remaining) {
                chunk = remaining;
            }
            noc.async_write(
                src, acc, chunk, {.offset_bytes = src_off}, {.page_id = page, .offset_bytes = off});
            page += 1;
            off = 0;
            src_off += chunk;
            remaining -= chunk;  // chunk <= remaining, so this cannot underflow
        }
    }
}

}  // namespace ttdm

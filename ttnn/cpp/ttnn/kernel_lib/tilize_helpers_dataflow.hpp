// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

/**
 * @brief Controls CB page granularity for tilize dataflow
 *
 * Determines how the reader pushes data into the input CB for the
 * compute_kernel_lib::tilize helper (ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp).
 *
 * TILE: CB page_size = tile_size. Reader pushes width_in_tiles pages per
 *       tile-height block. Matches compute_kernel_lib::tilize symmetric mode:
 *         compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks);
 *
 * ROW:  CB page_size = padded_row_bytes (one stick = one page). Reader pushes
 *       1 page per row. Matches compute_kernel_lib::tilize asymmetric mode:
 *         compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks, total_num_rows);
 *       Finer granularity — compute can start as soon as 32 rows arrive.
 *       When total_num_rows < 32, this also reduces L1 usage for the CB
 *       since only total_num_rows row-pages need to be buffered instead
 *       of width_in_tiles tile-pages (which always assume 32 rows of data).
 *
 * Example — TILE granularity (reader kernel):
 *   dataflow_kernel_lib::read_sticks_for_tilize<cb_in>(accessor, num_rows, row_bytes);
 *
 * Example — ROW granularity (reader kernel):
 *   dataflow_kernel_lib::read_sticks_for_tilize<cb_in, dataflow_kernel_lib::TilizeGranularity::ROW>(
 *       accessor, num_rows, row_bytes);
 *
 * Corresponding compute kernel for TILE:
 *   compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks);
 *
 * Corresponding compute kernel for ROW:
 *   compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks, total_num_rows);
 */
enum class TilizeGranularity : uint8_t {
    TILE,
    ROW,
};

/**
 * @brief How the per-row stick reads program the NoC command buffer.
 *
 * Generic (default): one `noc_async_read` per row, each re-programming the full
 *   NoC command (coordinate + local address + size) and re-running
 *   `TensorAccessor::get_noc_addr` (a rank loop plus a divide/modulo by the bank
 *   count) for every row.
 *
 * Stateful: `noc_async_read_set_state` / `noc_async_read_with_state`
 *   (dataflow_api.h:673,708). `set_state` pins the NoC *coordinate*, so the rows
 *   are visited **bank-major**: for an interleaved tensor, pages `p` and
 *   `p + num_banks` live in the same bank exactly one aligned page apart, so one
 *   armed command covers every `num_banks`-th row and its source address
 *   advances by a constant. That also removes all but one address computation
 *   per bank. Falls back to Generic (same call, no extra cost) when the accessor
 *   is not interleaved or when there are fewer than 2 rows per bank (nothing to
 *   amortize). Deliberately NOT the `one_packet` flavour of the same API — that
 *   one hangs every core on a watcher build; see the note in the .inl.
 *
 * Only affects TILE granularity; ROW granularity reads one row per CB page and
 * has no group to amortize over.
 */
enum class StickReadMode : uint8_t {
    Generic,
    Stateful,
};

/**
 * @brief Read row-major sticks from DRAM into a CB for tilization
 *
 * Reads total_num_rows sticks, grouping them into tile-height blocks.
 * Handles non-tile-aligned widths by padding the L1 stride.
 * Handles non-tile-aligned heights by pushing full tile pages for the
 * last partial block (untouched rows contain stale data).
 *
 * With TILE granularity (default):
 *   - CB must be configured with page_size = tile_size
 *   - Pushes width_in_tiles pages per block
 *   - Compute side: compute_kernel_lib::tilize<W, cb_in, cb_out>(num_blocks)
 *   - CB sizing: double_buffer * width_in_tiles * tile_size
 *
 * With ROW granularity:
 *   - CB must be configured with page_size = padded_row_bytes
 *   - Pushes 1 page per row
 *   - Compute side: compute_kernel_lib::tilize<W, cb_in, cb_out>(num_blocks, total_num_rows)
 *   - CB sizing: double_buffer * min(tile_h, total_num_rows) * padded_row_bytes
 *     (can be smaller than TILE mode when total_num_rows < tile_h)
 *
 * @tparam cb_id Circular buffer to write into (must be constexpr)
 * @tparam granularity TILE (default) or ROW
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the source tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to read
 * @param row_bytes Actual bytes per stick to read (may be non-tile-aligned).
 *        Combined with byte_offset_within_page this selects a chunk along W:
 *        each stick contributes `row_bytes` bytes starting at byte
 *        `byte_offset_within_page` of its source page.
 * @param start_page Starting page/stick index in the accessor (default 0).
 *        For multi-core work distribution, pass the per-core start_row offset
 *        so the accessor reads from the correct tensor slice.
 * @param byte_offset_within_page Byte offset inside each source page where
 *        reading begins (default 0). Used for wide-W chunking: wrap this
 *        helper in a chunk-outer loop and pass
 *        `byte_offset_within_page = chunk_id * row_bytes` so each call reads a
 *        different W-chunk of the same set of sticks. CB sizing (per call)
 *        then scales with `row_bytes` (the chunk width), not the full row,
 *        bounding L1 footprint regardless of total W.
 */
template <
    uint32_t cb_id,
    TilizeGranularity granularity = TilizeGranularity::TILE,
    StickReadMode read_mode = StickReadMode::Generic,
    typename Accessor>
FORCE_INLINE void read_sticks_for_tilize(
    const Accessor& accessor,
    uint32_t total_num_rows,
    uint32_t row_bytes,
    uint32_t start_page = 0,
    uint32_t byte_offset_within_page = 0);

/**
 * @brief Read one tile-height band of row-major sticks into a CALLER-OWNED L1 region
 *
 * The inner row loop of read_sticks_for_tilize()'s TILE mode, exposed on its own
 * so that (a) the caller can own the CB handshake, and (b) the band can be split
 * across both data-movement RISC-Vs.
 *
 * Use this instead of read_sticks_for_tilize() when the reads of ONE block are
 * shared between NCRISC and BRISC (the "split reader" pattern): a circular buffer
 * must have exactly one producer, so only one RISC-V may call
 * `cb_reserve_back`/`cb_push_back` on it — the other must be handed the reserved
 * L1 window and write into it, which this helper does. It issues no CB call and
 * no `noc_async_read_barrier()`; the caller owns both (each RISC-V barriers its
 * own reads).
 *
 * The band is partitioned by GROUP, not by contiguous row range: with
 * `num_splits = 2`, split 0 takes groups 0, 2, 4, ... and split 1 takes
 * groups 1, 3, 5, .... Under StickReadMode::Stateful a group is a whole bank's
 * worth of rows, so each half keeps ~num_banks/2 armed commands with several
 * reads each — a contiguous row split would instead halve the rows per bank and
 * give both halves the same number of arms. Under Generic a group is one row, so
 * the two halves interleave rows.
 *
 * @tparam read_mode Generic or Stateful (see StickReadMode)
 * @tparam num_splits How many DM RISC-Vs share this band (1 = no split)
 * @param accessor TensorAccessor for the source tensor (stick-indexed)
 * @param first_page Page/stick index of row 0 of this band
 * @param row_bytes Bytes to read per row
 * @param byte_offset_within_page Byte offset inside each source page
 * @param l1_addr Destination L1 address of row 0 (inside the reserved CB window)
 * @param l1_row_stride Bytes between consecutive rows in L1 (padded row bytes)
 * @param num_rows Rows in this band (<= tile height)
 * @param split_id Which of the num_splits DM RISC-Vs this call is (0-based)
 */
template <StickReadMode read_mode = StickReadMode::Generic, uint32_t num_splits = 1, typename Accessor>
FORCE_INLINE void read_stick_rows_for_tilize(
    const Accessor& accessor,
    uint32_t first_page,
    uint32_t row_bytes,
    uint32_t byte_offset_within_page,
    uint32_t l1_addr,
    uint32_t l1_row_stride,
    uint32_t num_rows,
    uint32_t split_id = 0);

/**
 * @brief Write untilized sticks from a CB to DRAM
 *
 * Reads total_num_rows worth of untilized data from the CB (produced by
 * the compute_kernel_lib::untilize helper from untilize_helpers.hpp) and
 * writes the valid sticks to DRAM.
 *
 * Handles non-tile-aligned widths by skipping L1 padding between rows.
 * Handles non-tile-aligned heights by popping full tile pages for the
 * last partial block but only writing the valid rows.
 *
 * Always operates at TILE granularity — the compute_kernel_lib::untilize
 * helper always produces tile-sized pages on its output CB.
 *
 * Corresponding compute kernel:
 *   compute_kernel_lib::untilize<width_tiles, cb_in, cb_out>(num_blocks);
 *
 * @tparam cb_id Circular buffer to read from (must be constexpr)
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the destination tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to write
 * @param row_bytes Actual bytes per stick to write (may be non-tile-aligned).
 *        Combined with byte_offset_within_page this selects a chunk along W:
 *        each stick writes `row_bytes` bytes starting at byte
 *        `byte_offset_within_page` of its destination page.
 * @param start_page Starting page/stick index in the accessor (default 0).
 *        For multi-core work distribution, pass the per-core start_row offset
 *        so the accessor writes to the correct tensor slice.
 * @param byte_offset_within_page Byte offset inside each destination page
 *        where writing begins (default 0). Symmetric to the read helper's
 *        parameter: wrap this helper in a chunk-outer loop and pass
 *        `byte_offset_within_page = chunk_id * row_bytes` so each call writes
 *        a different W-chunk of the same set of sticks.
 */
template <uint32_t cb_id, typename Accessor>
FORCE_INLINE void write_sticks_after_untilize(
    const Accessor& accessor,
    uint32_t total_num_rows,
    uint32_t row_bytes,
    uint32_t start_page = 0,
    uint32_t byte_offset_within_page = 0);

/**
 * @brief Bank-address table for a run of tilize read bands, built ONCE per core
 *
 * `read_stick_rows_for_tilize<StickReadMode::Stateful>` already reads bank-major
 * with an armed command buffer, but it rebuilds its bank addresses on every call:
 * one `accessor.get_noc_addr` per bank group per BAND. On a plan with several
 * bands (tile-row blocks) per core that address generation is the residual — it
 * is pure NCRISC work, and on a read-bound plan every NCRISC cycle is on the
 * critical path. A per-RISC Tracy timeline of tilize's `alias_out` crossover
 * (`[1,1,2048,512]` -> BLOCK-sharded, 64 cores, 8 bands/core) attributes
 * **4 857 of 17 017 NCRISC cycles to address generation alone**, with the unpack
 * TRISC blocked in `cb_wait_front` for 90 % of the kernel.
 *
 * This class removes all but `num_banks` of those calls. It rests on the
 * interleaved page mapping being an affine function of the page index: page `p`
 * lives in bank `p % num_banks` at bank-page `p / num_banks`
 * (`dataflow_api_addrgen.h` InterleavedAddrGen), so for a fixed byte offset
 *
 *     addr(first_page + m) = table[m % num_banks]
 *                          + (m / num_banks) * aligned_page_size
 *
 * for EVERY m >= 0 — the table primed at `first_page` therefore serves every band
 * at `first_page + k * stride`, for any k and any stride. `read_band()` walks the
 * bands the same bank-major way the Stateful mode does (one arm per bank group,
 * then a running `+= aligned_page` inside the group), so it keeps that mode's
 * per-read cost and only drops the address generation.
 *
 * Applicable only to an INTERLEAVED accessor (a sharded one has no such affine
 * map) and only worth priming when there is more than one band per core and at
 * least two rows per bank group — otherwise the priming pass costs more than the
 * calls it replaces. Both are compile-time checkable by the caller; `prime()`
 * asserts the accessor part.
 *
 * @tparam num_banks Bank period — `NUM_DRAM_BANKS` or `NUM_L1_BANKS`.
 *
 * @code
 *   dataflow_kernel_lib::InterleavedStickBands<NUM_DRAM_BANKS> bands;
 *   bands.prime(accessor, start_row, byte_offset);           // num_banks accessor calls, once
 *   for (uint32_t b = 0; b < num_blocks; ++b) {
 *       cb_reserve_back(cb_in, chunk_wt);
 *       bands.read_band(b * 32, row_bytes, get_write_ptr(cb_in), row_bytes, 32);
 *       noc_async_read_barrier();
 *       cb_push_back(cb_in, chunk_wt);
 *   }
 * @endcode
 */
template <uint32_t num_banks>
class InterleavedStickBands {
public:
    /**
     * @brief Build the bank table. Costs `num_banks` accessor calls, once per core.
     * @param accessor Interleaved TensorAccessor for the source tensor
     * @param first_page Page index the table is anchored at (band offsets are relative to it)
     * @param byte_offset_within_page Byte offset inside each source page (folded into the table)
     */
    template <typename Accessor>
    FORCE_INLINE void prime(const Accessor& accessor, uint32_t first_page, uint32_t byte_offset_within_page);

    /**
     * @brief Read one band of `num_rows` sticks, bank-major, with no accessor call
     * @param page_offset Band start, RELATIVE to the `first_page` given to prime()
     * @param row_bytes Bytes to read per row
     * @param l1_addr Destination L1 address of row 0 of the band
     * @param l1_row_stride Bytes between consecutive rows in L1
     * @param num_rows Rows in this band
     */
    FORCE_INLINE void read_band(
        uint32_t page_offset, uint32_t row_bytes, uint32_t l1_addr, uint32_t l1_row_stride, uint32_t num_rows) const;

    /**
     * @brief The address `read_band` would use for row `page_offset` of the run
     *
     * Exposed so a caller can re-derive it through the accessor under `ASSERT`
     * and check the affine identity on device instead of assuming it.
     */
    FORCE_INLINE uint64_t addr_of(uint32_t page_offset) const;

private:
    uint64_t noc_addr_[num_banks];
    uint32_t aligned_page_ = 0;
};

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl"

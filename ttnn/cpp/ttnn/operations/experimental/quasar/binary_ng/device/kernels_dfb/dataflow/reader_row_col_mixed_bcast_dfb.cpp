// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for binary_ng's MIXED (ROW_A_COL_B / ROW_B_COL_A) subtile
// broadcast -- BOTH operands broadcast a DIFFERENT subtile axis at once (one logical row, one logical
// column). Port of the CircularBuffer kernels_ng/dataflow/reader_interleaved_row_col_mixed_bcast.cpp with
// the CB->DFB API swap (mirroring reader_col_bcast_dfb.cpp / reader_bcast_dfb.cpp: get_arg(args::...),
// dfb::in0/in1, get_entry_size(), TensorAccessor(tensor::in0/in1)).
//
// HYBRID broadcast: the COL operand is expanded HERE (reader software-fill FILL_TILE_WITH_FIRST_COLUMN,
// UNCONDITIONAL -- NOT gated by BCAST_LLK), delivered once per tile-row and reused across the row; the ROW
// operand is delivered as a raw PARTIAL tile (BCAST_LLK path, no fill) and the compute expands it via
// unary_bcast<ROW>. This is the FIRST DFB consumer of reader-side software-fill (DataflowBuffer::
// get_write_ptr() + fill_tile_utils.hpp); the six single-operand subtile types all used the pure-LLK,
// no-fill reader path. The split is a deliberate reader/compute load-balance (see the compute kernel).
//
// Tile walk (both operands are broadcast-style, so BOTH omit the start_th*Wt term and the per-th += Wt
// advance, and both advance the c-dim by c_stride directly -- re-reading their single row / column of
// tiles): the COL operand reads ONE tile per tile-row (page = tile_offset + th, OUTSIDE the tw loop) and
// software-fills it; the ROW operand reads Wt tiles across the row (page = tile_offset + tw, INSIDE the tw
// loop) as raw partial tiles. SRC_BCAST_COL (1 -> a is the COL operand, 0 -> b is) and SRC_BCAST_ROW_B
// (1 -> b is the ROW operand, 0 -> a is) select the two roles; the factory sets them equal (both 1 for
// ROW_B_COL_A, both 0 for ROW_A_COL_B). A bcast operand's shape can never match the output shard spec, so
// this factory never borrows either operand on the broadcast path -- both take the interleaved (NoC) path
// (SRC_SHARDED / SRC_SHARDED_B are 0 here).

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
// Reader software-fill helpers (FILL_TILE_WITH_FIRST_COLUMN / _B map to fill_tile_with_first_column* per
// dtype via make_dataflow_defines). Included by full repo path exactly as the kernels_ng mixed reader does
// (the JIT compile has the ttnn root on its include path); the single-operand DFB readers did not need it.
#include "ttnn/operations/experimental/quasar/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t src_num_tiles = get_arg(args::src_num_tiles);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t nD_stride = get_arg(args::nD_stride);
    const uint32_t d_stride = get_arg(args::d_stride);
    const uint32_t n_stride = get_arg(args::n_stride);
    const uint32_t c_stride = get_arg(args::c_stride);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);  // collapsed dims > 5
    const uint32_t nD_stride_b = get_arg(args::nD_stride_b);
    const uint32_t d_stride_b = get_arg(args::d_stride_b);
    const uint32_t n_stride_b = get_arg(args::n_stride_b);
    const uint32_t c_stride_b = get_arg(args::c_stride_b);
    const uint32_t src_num_tiles_b = get_arg(args::src_num_tiles_b);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    // Both operands are read tile-by-tile over the NoC (a bcast operand is never borrowed on this path).
#if !SRC_SHARDED
    const uint32_t src_tile_bytes = dfb_in0.get_entry_size();
    const auto src = TensorAccessor(tensor::in0);
#endif
#if !SRC_SHARDED_B
    const uint32_t src_tile_bytes_b = dfb_in1.get_entry_size();
    const auto src_b = TensorAccessor(tensor::in1);
#endif
    constexpr uint32_t onetile = 1;
    // HAS_SHARDING (factory-set) mirrors the CB kernel's sharded-output row handling for an interleaved
    // operand: when set, each tile row spans only dst_shard_width tiles before wrapping.
    constexpr bool has_sharding = HAS_SHARDING;
    const uint32_t HtWt = Ht * Wt;

    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    const uint32_t offset_nd = start_tile_id % tiles_per_nd;
    const uint32_t offset_d = offset_nd % tiles_per_d;
    const uint32_t offset_n = offset_d % tiles_per_n;
    const uint32_t offset_c = offset_n % HtWt;
    uint32_t start_nd = start_tile_id / tiles_per_nd;
    uint32_t start_d = offset_nd / tiles_per_d;
    uint32_t start_n = offset_d / tiles_per_n;
    uint32_t start_c = offset_n / HtWt;
    uint32_t start_th = offset_c / Wt;
    uint32_t start_tw = offset_c % Wt;
    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

    // INPUT tile offsets. BOTH operands are broadcast-style (COL reuses its column of tiles across the
    // width; ROW reuses its row of tiles across the height), so neither offset includes start_th*Wt and
    // neither advances by Wt per th; both advance the c-dim by c_stride directly (re-reading their single
    // row / column). The outer (n/d/nd) advances use the standard next_*_shift.
    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        // COL operand: read/publish ONE tile for this tile-row (page keyed by th) and
                        // software-fill it (FILL_TILE_WITH_FIRST_COLUMN, unconditional -- this is the COL
                        // broadcast). Reused across the row by the compute's freq loop.
                        //
                        // COHERENCE (Quasar): a plain cacheable RISC store fill does NOT work here. Quasar DM
                        // cores have a write-back L1 D$ (per-core) + shared L2 that are INCOHERENT with TL1
                        // (the SRAM the compute consumer + the NOC engine read): a cacheable fill stays dirty
                        // in the DM D$ -- invisible to the consumer (it reads stale TL1) AND later evicted
                        // over the neighbor llk_post DFB, corrupting it. So the fill MUST use a coherent store.
                        // Here we write through the NON-CACHEABLE L1 alias (MEM_L1_UNCACHED_BASE): the stores
                        // (and the col-0 reads) bypass the D$/L2 and go straight to TL1, so the consumer sees
                        // the fill and no dirty line can clobber a neighbor -- no flush needed. (An L2 flush,
                        // flush_l2_cache_range(get_write_ptr(), get_entry_size()) before push_back, is the
                        // equivalent alternative -- cf. ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl.)
                        // WH/BH DFB is CB-backed shared L1 with no incoherent write-back D$, so the plain fill
                        // is already visible there -- the alias offset is 0 (compiled out).
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
                        const uint32_t col_uncached_off = MEM_L1_UNCACHED_BASE;
#else
                        const uint32_t col_uncached_off = 0;
#endif
#if SRC_BCAST_COL  // a (in0) is the COL operand
                        dfb_in0.reserve_back(onetile);
#if !SRC_SHARDED
                        noc.async_read(
                            src, dfb_in0, src_tile_bytes, {.page_id = tile_offset + th}, {.offset_bytes = 0});
                        noc.async_read_barrier();
                        FILL_TILE_WITH_FIRST_COLUMN(dfb_in0.get_write_ptr() + col_uncached_off);
#endif
                        dfb_in0.push_back(onetile);
#else  // b (in1) is the COL operand
                        dfb_in1.reserve_back(onetile);
#if !SRC_SHARDED_B
                        noc.async_read(
                            src_b, dfb_in1, src_tile_bytes_b, {.page_id = tile_offset_b + th}, {.offset_bytes = 0});
                        noc.async_read_barrier();
                        FILL_TILE_WITH_FIRST_COLUMN_B(dfb_in1.get_write_ptr() + col_uncached_off);
#endif
                        dfb_in1.push_back(onetile);
#endif
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            // ROW operand: read Wt raw PARTIAL tiles across the row (page keyed by tw).
                            // BCAST_LLK=1 -> no software fill; the compute's unary_bcast<ROW> expands it.
#if SRC_BCAST_ROW_B  // b (in1) is the ROW operand
                            dfb_in1.reserve_back(onetile);
#if !SRC_SHARDED_B
                            noc.async_read(
                                src_b, dfb_in1, src_tile_bytes_b, {.page_id = tile_offset_b + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
                            dfb_in1.push_back(onetile);
#endif
#else  // a (in0) is the ROW operand
                            dfb_in0.reserve_back(onetile);
#if !SRC_SHARDED
                            noc.async_read(
                                src, dfb_in0, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
                            dfb_in0.push_back(onetile);
#endif
#endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
                    }
                    // Per-c: BOTH operands advance by c_stride directly (re-read their row / column).
#if !SRC_SHARDED
                    tile_offset += c_stride;
#endif
#if !SRC_SHARDED_B
                    tile_offset_b += c_stride_b;
#endif
                }
#if !SRC_SHARDED
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_B
                tile_offset_b += next_n_shift_b;
#endif
            }
#if !SRC_SHARDED
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_B
            tile_offset_b += next_d_shift_b;
#endif
        }
#if !SRC_SHARDED
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_B
        tile_offset_b += next_nd_shift_b;
#endif
    }
}

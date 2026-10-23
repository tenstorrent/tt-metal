// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for binary_ng's SCALAR subtile-broadcast binary op.
//
// Port of the CircularBuffer kernels_ng/dataflow/reader_interleaved_scalar_bcast.cpp with the CB->DFB API
// swap (mirroring reader_col_bcast_dfb.cpp / reader_bcast_dfb.cpp: get_arg(args::...), dfb::in0/in1,
// get_entry_size(), TensorAccessor(tensor::in0/in1), HAS_SHARDING). SCALAR differs from the COL reader in
// the broadcast operand's tile walk: its single element (logical tile [1,1]) is reused across the ENTIRE
// (N,C) slab -- Ht * Wt output tiles -- so the broadcast operand reads ONE tile PER (N,C) SLAB (page_id =
// tile_offset, OUTSIDE both the th and tw loops), while the other operand reads Ht * Wt tiles per slab
// (page_id = tile_offset(_b) + tw, inside the tw loop, same as the no-bcast / COL reader). The broadcast
// operand's tile offset does NOT advance by Wt per th (no per-th += Wt), and its c-dim advance uses
// c_stride directly (re-reading the same single tile across N/C) -- the same closing advance COL uses, just
// applied one loop level higher since there is no per-th fetch to interleave it with.
//
// LLK broadcast path (BCAST_LLK=1): the reader delivers the single-element tile only (the broadcast
// operand's tile carries one valid element); it does NOT fill the tile -- the compute's
// unary_bcast<SCALAR> expands that single element across the whole tile (no software FILL_TILE here).
// Each operand is still independently SHARDED (borrow its resident L1 shard) or INTERLEAVED (NoC-read via
// TensorAccessor(tensor::in0/in1)); this factory never borrows on the broadcast path (a bcast operand's
// shape can't match the output shard spec), so in practice both operands take the interleaved (NoC) path
// here.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

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

    // A SHARDED non-broadcast operand publishes its whole resident shard up front (no NoC). A SHARDED
    // broadcast operand instead publishes one tile per (N,C) slab inside the loop below (its single tile
    // is consumed once per slab, reused across the whole Ht * Wt block), so it is NOT bulk-published here.
#if SRC_SHARDED
#if !SRC_BCAST
    dfb_in0.reserve_back(src_num_tiles);
    dfb_in0.push_back(src_num_tiles);
#endif
#else
    const uint32_t src_tile_bytes = dfb_in0.get_entry_size();
    const auto src = TensorAccessor(tensor::in0);
#endif
#if SRC_SHARDED_B
#if !SRC_BCAST_B
    dfb_in1.reserve_back(src_num_tiles_b);
    dfb_in1.push_back(src_num_tiles_b);
#endif
#else
    const uint32_t src_tile_bytes_b = dfb_in1.get_entry_size();
    const auto src_b = TensorAccessor(tensor::in1);
#endif
    constexpr uint32_t onetile = 1;
    // HAS_SHARDING (factory-set) mirrors the CB kernel's sharded-output row handling for an
    // interleaved operand: when set, each tile row spans only dst_shard_width tiles before wrapping.
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

    // this is the INPUT tile offset. The broadcast operand's single tile is reused across the entire
    // (N,C) slab (its th-indexed tile is read once per (N,C) slab), so its offset omits the start_th*Wt
    // term (and the per-th += Wt below).
    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
#if !SRC_BCAST
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b;
#if !SRC_BCAST_B
    tile_offset_b += start_th * Wt;
#endif
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    // Broadcast operand: read/publish ONE tile for this ENTIRE (N,C) slab (page_id keyed
                    // by tile_offset alone, reused for every th/tw in the slab below). BCAST_LLK=1 ->
                    // deliver the single-element tile; the compute's unary_bcast<SCALAR> expands it across
                    // the whole tile (no software FILL_TILE here).
#if SRC_BCAST
                    dfb_in0.reserve_back(onetile);
#if !SRC_SHARDED
                    noc.async_read(src, dfb_in0, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                    noc.async_read_barrier();
#endif
                    dfb_in0.push_back(onetile);
#endif
#if SRC_BCAST_B
                    dfb_in1.reserve_back(onetile);
#if !SRC_SHARDED_B
                    noc.async_read(src_b, dfb_in1, src_tile_bytes_b, {.page_id = tile_offset_b}, {.offset_bytes = 0});
                    noc.async_read_barrier();
#endif
                    dfb_in1.push_back(onetile);
#endif
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            // Non-broadcast operand: read Wt tiles across the row (page_id keyed by tw).
#if !SRC_BCAST && !SRC_SHARDED
                            dfb_in0.reserve_back(onetile);
                            noc.async_read(
                                src, dfb_in0, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
                            dfb_in0.push_back(onetile);
#endif
#if !SRC_BCAST_B && !SRC_SHARDED_B
                            dfb_in1.reserve_back(onetile);
                            noc.async_read(
                                src_b, dfb_in1, src_tile_bytes_b, {.page_id = tile_offset_b + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
                            dfb_in1.push_back(onetile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
#if !SRC_BCAST && !SRC_SHARDED
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_B && !SRC_SHARDED_B
                        tile_offset_b += Wt;
#endif
                    }
#if !SRC_SHARDED
#if SRC_BCAST
                    // broadcast operand: advance by c_stride directly (re-reads the same single tile)
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#endif
#if !SRC_SHARDED_B
#if SRC_BCAST_B
                    tile_offset_b += c_stride_b;
#else
                    tile_offset_b += next_c_shift_b;
#endif
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

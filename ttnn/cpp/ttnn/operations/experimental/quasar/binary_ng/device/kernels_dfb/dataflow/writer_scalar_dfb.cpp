// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for binary_ng's tensor-SCALAR op.
//
// Copy of writer_no_bcast_dfb.cpp with one addition: this writer is ALSO the producer of the RHS
// input DFB (`in1`) and fills it ONCE with the packed scalar before draining the output. The reader
// produces `in0` only; the compute waits on `in1` once and reuses tile index 0. The output-drain body
// (DST_SHARDED credit drain / interleaved NoC write) is byte-for-byte the no-broadcast writer's.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
// Scalar-fill helpers (FILL_WITH_VALUE / FILL_WITH_VALUE_FLOAT map to fill_with_val* per dtype via
// make_dataflow_defines). Included by full repo path exactly as the CB writer_interleaved_scalar.cpp
// does (the JIT compile has the ttnn root on its include path).
#include "ttnn/operations/experimental/quasar/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const uint32_t packed_scalar = get_arg(args::packed_scalar);  // NEW writer arg (scalar RHS)
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);  // collapsed dims > 5

    constexpr uint32_t onetile = 1;

    // Fill the RHS scalar tile ONCE (coherent uncached-L1-alias store on Quasar DM cores; offset 0
    // elsewhere). A plain cacheable RISC store fill does NOT work on Quasar: the DM core's write-back
    // L1 D$ + shared L2 are INCOHERENT with the TL1 SRAM the compute consumer + NOC engine read, so a
    // cacheable fill stays dirty in the D$ -- invisible to the consumer AND later evicted over the
    // neighbor DFB, corrupting it. Writing through the NON-CACHEABLE L1 alias (MEM_L1_UNCACHED_BASE)
    // bypasses the D$/L2 and goes straight to TL1, so the consumer sees the fill and no dirty line can
    // clobber a neighbor -- no flush needed. WH/BH DFB is CB-backed shared L1 with no incoherent
    // write-back D$, so the plain fill is already visible there -- the alias offset is 0 (compiled out).
    // See reader_row_col_mixed_bcast_dfb.cpp for the full D$/TL1 incoherence rationale.
    DataflowBuffer dfb_in1(dfb::in1);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    const uint32_t scalar_uncached_off = MEM_L1_UNCACHED_BASE;
#else
    const uint32_t scalar_uncached_off = 0;
#endif
    dfb_in1.reserve_back(onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    const auto float_ptr = reinterpret_cast<const float*>(&packed_scalar);
    FILL_WITH_VALUE_FLOAT(dfb_in1.get_write_ptr() + scalar_uncached_off, *float_ptr);
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(dfb_in1.get_write_ptr() + scalar_uncached_off, packed_scalar);
#endif
    dfb_in1.push_back(onetile);

    DataflowBuffer dfb_out(dfb::out);

#if DST_SHARDED
    // Output shard is written in place by the compute kernel; just drain the ring credits.
    dfb_out.wait_front(dst_num_tiles);
    dfb_out.pop_front(dst_num_tiles);
#else
    Noc noc;
    const uint32_t dst_tile_bytes = dfb_out.get_entry_size();
    const auto dst = TensorAccessor(tensor::out);

    // HAS_SHARDING (factory-set) mirrors the CB writer's sharded-output row handling.
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

    uint32_t num_tiles_written = 0;
    uint32_t dst_tile_offset = start_tile_id;

    for (uint32_t nd = start_nd; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_written < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_written < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
                             ++tw, ++num_tiles_written) {
                            //  write a tile to dst; for a full dst the tile offset grows linearly
                            dfb_out.wait_front(onetile);
                            noc.async_write(
                                dfb_out, dst, dst_tile_bytes, {}, {.page_id = dst_tile_offset + num_tiles_written});
                            noc.async_write_barrier();
                            dfb_out.pop_front(onetile);
                        }
                        if constexpr (has_sharding) {
                            // adjust the output tile offset since we had to skip parts of the row
                            dst_tile_offset += (Wt - dst_shard_width);
                        } else {
                            // otherwise, next row of tiles should start at the first column
                            start_tw = 0;
                        }
                    }
                }
            }
        }
    }
#endif
}

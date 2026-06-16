// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of kernels/dataflow/writer_interleaved_scalar.cpp.
// Bound by BinaryNgDeviceOperation::ProgramSpecFactory on the
//   no-broadcast x tile x FPU x scalar-b (no tensor b) x interleaved path.
// This writer both fills the scalar tile into the rhs CB (once) and writes the output tiles.
// Logic is unchanged from the legacy writer; only the resource-access mechanism is converted to
// Metal 2.0 named bindings:
//   - CB ids c_1 / c_2 -> dfb::src_b (scalar) / dfb::dst (output)
//   - dst address + TensorAccessorArgs -> ta::dst (address auto-injected)
//   - positional get_arg_val / get_compile_time_arg_val -> get_arg(args::...)
// The #if DST_SHARDED path is preserved verbatim; the ProgramSpecFactory only selects this kernel
// on the interleaved (unsharded) path, so DST_SHARDED is compiled out, but it is kept for
// faithfulness to the source. The packed scalar arrives as the named runtime arg `packed_scalar`
// (a value, not a pointer), matching the legacy positional slot 0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const uint32_t packed_scalar = get_arg(args::packed_scalar);
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);  // collapsed dims > 5
    const uint32_t HtWt = Ht * Wt;

    constexpr uint32_t onetile = 1;

    Noc noc;
    CircularBuffer cb_src(dfb::src_b);
    CircularBuffer cb_dst(dfb::dst);

    // we only need to fill a tile with the scalar value once
    cb_src.reserve_back(onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    const auto float_ptr = reinterpret_cast<const float*>(&packed_scalar);
    FILL_WITH_VALUE_FLOAT(cb_src.get_write_ptr(), *float_ptr);
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_src.get_write_ptr(), packed_scalar);
#endif
    cb_src.push_back(onetile);

#if !DST_SHARDED
    const uint32_t dst_tile_bytes = cb_dst.get_tile_size();
    const auto dst = TensorAccessor(ta::dst);
    constexpr bool has_sharding = get_arg(args::has_sharding) == 1;

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
                            // write a tile to dst
                            cb_dst.wait_front(onetile);
                            noc.async_write(
                                cb_dst, dst, dst_tile_bytes, {}, {.page_id = dst_tile_offset + num_tiles_written});
                            noc.async_write_barrier();
                            cb_dst.pop_front(onetile);
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

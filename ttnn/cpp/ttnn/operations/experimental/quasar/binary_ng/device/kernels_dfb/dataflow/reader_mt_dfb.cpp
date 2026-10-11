// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-thread, implicit-sync DFB reader for binary_ng: feeds one in0 and one in1 entry per output tile.
// Thread t of N takes the core's output tiles t, t + N, ...; the strided DFBs hand them to compute in
// order. Every transfer is one TXN_ID read of a full tile:
//   - a plain operand tile is read straight from its tensor;
//   - a broadcast operand tile (A_FILL / B_FILL: 1 first element, 2 first row, 3 first column) is read
//     into this thread's scratch tile, expanded by the CPU, flushed from the L2 cache, and then read
//     locally into the DFB; it is re-expanded only when the source tile changes;
//   - a scalar operand (B_SCALAR) is a scratch tile filled once with the packed scalar.
// The host pads the tile count to num_padded_tiles, a multiple of N, so every txn-ID batch is full: with
// two implicit DFBs, a partial tail batch on both would make each DFB's destructor wait for compute,
// which waits on the other DFB's unposted tail. Compute drops the padding tiles.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/scoped_lock_cache_ops.h"
#include "ttnn/operations/experimental/quasar/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

namespace {

struct Operand {
    uint32_t nD_stride;
    uint32_t d_stride;
    uint32_t n_stride;
    uint32_t c_stride;
    uint32_t Ht;
    uint32_t Wt;

    uint32_t tile_id(uint32_t nd, uint32_t d, uint32_t n, uint32_t c, uint32_t th, uint32_t tw) const {
        return nd * nD_stride + d * d_stride + n * n_stride + c * c_stride + (Ht > 1 ? th : 0) * Wt + (Wt > 1 ? tw : 0);
    }
};

template <uint32_t fill>
void expand_tile(uint32_t addr) {
    if constexpr (fill == 1) {
        fill_tile_with_first_element_bfloat16(addr);
    } else if constexpr (fill == 2) {
        fill_tile_with_first_row_bfloat16(addr);
    } else if constexpr (fill == 3) {
        fill_tile_with_first_column_bfloat16(addr);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t num_padded_tiles = get_arg(args::num_padded_tiles);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const Operand op_a{
        get_arg(args::nD_stride),
        get_arg(args::d_stride),
        get_arg(args::n_stride),
        get_arg(args::c_stride),
        get_arg(args::a_Ht),
        get_arg(args::a_Wt)};
#if !B_SCALAR
    const Operand op_b{
        get_arg(args::nD_stride_b),
        get_arg(args::d_stride_b),
        get_arg(args::n_stride_b),
        get_arg(args::c_stride_b),
        get_arg(args::b_Ht),
        get_arg(args::b_Wt)};
#endif

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const auto src_a = TensorAccessor(tensor::in0);
#if !B_SCALAR
    const auto src_b = TensorAccessor(tensor::in1);
#endif
    const uint32_t tile_bytes = dfb_in0.get_entry_size();

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

#if A_FILL || B_FILL || B_SCALAR
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];
    Scratchpad<uint32_t> scratch(scratch::pad);
    const uint32_t scratch_a = scratch.get_base_address() + thread_id * 2 * tile_bytes;
    const uint32_t scratch_b = scratch_a + tile_bytes;
    // Reads `tile` of `src` into `addr` once per new source tile and expands it in place.
    auto stage = [&](const auto& src, uint32_t tile, uint32_t addr, uint32_t& staged_tile, auto expand) {
        if (tile == staged_tile) {
            return;
        }
        CoreLocalMem<uint32_t> dst(addr);
        noc.async_read(src, dst, tile_bytes, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // The NoC wrote L1 under the CPU's cache: drop stale lines, expand, then flush for the NoC read.
        invalidate_l2_cache_range(addr, tile_bytes);
        expand(addr);
        flush_l2_cache_range(addr, tile_bytes);
        staged_tile = tile;
    };
    uint32_t staged_a = UINT32_MAX;
    uint32_t staged_b = UINT32_MAX;
#endif
#if B_SCALAR
    fill_with_val_bfloat16(scratch_b, get_arg(args::packed_scalar));
    flush_l2_cache_range(scratch_b, tile_bytes);
#endif

    const uint32_t HtWt = Ht * Wt;
    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;

    for (uint32_t k = thread_id; k < num_padded_tiles; k += num_threads) {
        // Padding tiles re-read the core's last real tile.
        const uint32_t g = start_tile_id + (k < num_tiles ? k : num_tiles - 1);
        const uint32_t nd = g / tiles_per_nd;
        const uint32_t d = (g % tiles_per_nd) / tiles_per_d;
        const uint32_t n = (g % tiles_per_d) / tiles_per_n;
        const uint32_t c = (g % tiles_per_n) / HtWt;
        const uint32_t th = (g % HtWt) / Wt;
        const uint32_t tw = g % Wt;

        const uint32_t tile_a = op_a.tile_id(nd, d, n, c, th, tw);
#if A_FILL
        stage(src_a, tile_a, scratch_a, staged_a, expand_tile<A_FILL>);
        noc.async_read<NocOptions::TXN_ID>(
            self_ep, dfb_in0, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = scratch_a}, {});
#else
        noc.async_read<NocOptions::TXN_ID>(src_a, dfb_in0, {.page_id = tile_a}, {});
#endif

#if B_SCALAR
        noc.async_read<NocOptions::TXN_ID>(
            self_ep, dfb_in1, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = scratch_b}, {});
#else
        const uint32_t tile_b = op_b.tile_id(nd, d, n, c, th, tw);
#if B_FILL
        stage(src_b, tile_b, scratch_b, staged_b, expand_tile<B_FILL>);
        noc.async_read<NocOptions::TXN_ID>(
            self_ep, dfb_in1, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = scratch_b}, {});
#else
        noc.async_read<NocOptions::TXN_ID>(src_b, dfb_in1, {.page_id = tile_b}, {});
#endif
#endif
    }
}

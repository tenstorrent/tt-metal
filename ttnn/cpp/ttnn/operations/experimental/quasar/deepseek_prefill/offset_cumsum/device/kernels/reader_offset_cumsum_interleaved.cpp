// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset Cumsum Kernel
//
// Computes every device's global dispatch offset into each expert's token buffer.
//
// The global offset for a device combines two components:
//   1. Local offset: the elementwise sum of the input rows before that device's own row
//   2. Expert region offset: exclusive prefix sum of tile-aligned total token counts within
//      each chip's expert group (experts_per_chip stride)
//
// Inputs:
//   - input [H, W]: UINT32 interleaved tensor of per-device expert histograms
//     (H = devices along cluster_axis, W = n_routed_experts). Produced by all_gather of each
//     device's masked_bincount output.
//
// Outputs:
//   - offsets         [1, W]: this device's global dispatch offsets
//   - totals          [1, W]: sum of all H input rows (total tokens per expert)
//   - expert_region   [1, W]: expert region offsets only (shared component — exclusive
//                             prefix sum of tile-aligned totals within each chip group)
//   - all_offsets     [H, W]: row r is device r's `offsets`. A run is the span of an expert's
//                             buffer that one source device writes; any device can size any run.
//                             Rows include expert_region, so run r's length is row[r+1] - row[r],
//                             and the last run's is totals + expert_region - row[H-1],
//                             NOT totals - row[H-1].
//
// Runtime args:
//   - src_addr, dst_offsets_addr, dst_totals_addr, dst_expert_region_addr, dst_all_offsets_addr
//   - row_idx: which row of the prefix sum this device keeps in `offsets`

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "internal/local_mem/address_access.h"
#include "api/tile_constants.h"
#include "experimental/kernel_args.h"
#include "ckernel.h"

// L1 address the CPU accesses a scratchpad through: on Quasar DM the uncached alias, as DataflowBuffer pointers are,
// since the CPU caches are not coherent with NoC writes to L1; on WH/BH the plain address.
template <typename T>
uint32_t cpu_l1_addr(const Scratchpad<T>& scratch) {
    uintptr_t addr = local_mem::detail::AddressAccess::address(scratch.local_mem());
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    addr += MEM_L1_UNCACHED_BASE;
#endif
    return static_cast<uint32_t>(addr);
}

// The kernel is bound by scalar L1 accesses, not NOC traffic. Both helpers walk N columns at a time so
// the running sums stay in registers while each row is still read contiguously.

// sums[i0 + j] = sum over the H rows of in[.][i0 + j]
template <uint32_t N, uint32_t H, uint32_t in_stride>
FORCE_INLINE void sum_columns(const uint32_t* in, uint32_t* sums, uint32_t i0) {
    uint32_t acc[N] = {};
    for (uint32_t h = 0; h < H; h++) {
        const uint32_t* src = in + h * in_stride + i0;
#pragma GCC unroll N
        for (uint32_t j = 0; j < N; j++) {
            acc[j] += src[j];
        }
    }
#pragma GCC unroll N
    for (uint32_t j = 0; j < N; j++) {
        sums[i0 + j] = acc[j];
    }
}

// rows[r][i0 + j] = base[i0 + j] + sum over rows 0..r-1 of in[.][i0 + j]
template <uint32_t N, uint32_t H, uint32_t in_stride, uint32_t row_stride>
FORCE_INLINE void emit_rows(const uint32_t* in, const uint32_t* base, uint32_t* rows, uint32_t i0) {
    uint32_t acc[N];
#pragma GCC unroll N
    for (uint32_t j = 0; j < N; j++) {
        acc[j] = base[i0 + j];
    }
    for (uint32_t r = 0; r < H; r++) {
        const uint32_t* src = in + r * in_stride + i0;
        uint32_t* dst = rows + r * row_stride + i0;
#pragma GCC unroll N
        for (uint32_t j = 0; j < N; j++) {
            dst[j] = acc[j];
            acc[j] += src[j];
        }
    }
}

void kernel_main() {
    uint32_t row_idx = get_arg(args::row_idx);

    Noc noc;
    Scratchpad<uint32_t> in0(scratch::in0);
    Scratchpad<uint32_t> out0(scratch::out0);
    Scratchpad<uint32_t> rows_scratch(scratch::rows);
    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t offsets_page_size = get_arg(args::offsets_page_size);
    constexpr uint32_t totals_page_size = get_arg(args::totals_page_size);
    constexpr uint32_t expert_region_page_size = get_arg(args::expert_region_page_size);
    constexpr uint32_t all_offsets_page_size = get_arg(args::all_offsets_page_size);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t experts_per_chip = get_arg(args::experts_per_chip);

    const auto src_accessor = TensorAccessor(tensor::src);
    const auto dst_offsets_accessor = TensorAccessor(tensor::dst_offsets);
    const auto dst_totals_accessor = TensorAccessor(tensor::dst_totals);
    const auto dst_expert_region_accessor = TensorAccessor(tensor::dst_expert_region);
    const auto dst_all_offsets_accessor = TensorAccessor(tensor::dst_all_offsets);

    // These scratchpads are plain L1 scratch: everything is staged and sent from their base.

    constexpr uint32_t COLS = 8;
    constexpr uint32_t W_blocked = W / COLS * COLS;

    // All H input rows, at input_page_size stride, so the input is read from DRAM once.
    constexpr uint32_t in_stride = input_page_size / sizeof(uint32_t);
    // Plain (non-volatile) L1 views: the input arrives by DMA and is made visible by invalidate_l1_cache after
    // the read barrier; outputs are fenced before the NoC reads them. Volatile accesses cost several cycles each.
    const uint32_t* in = reinterpret_cast<const uint32_t*>(cpu_l1_addr(in0));
    for (uint32_t h = 0; h < H; h++) {
        noc.async_read(src_accessor, in0, input_page_size, {.page_id = h}, {.offset_bytes = h * input_page_size});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();

    // running_sum holds totals, then becomes the expert region offsets
    uint32_t* running_sum = reinterpret_cast<uint32_t*>(cpu_l1_addr(out0));
    for (uint32_t i0 = 0; i0 < W_blocked; i0 += COLS) {
        sum_columns<COLS, H, in_stride>(in, running_sum, i0);
    }
    for (uint32_t i0 = W_blocked; i0 < W; i0++) {
        sum_columns<1, H, in_stride>(in, running_sum, i0);
    }

    // 1. Write totals to DRAM. The barrier is required: step 2 overwrites running_sum in place.
    // A store can retire before it lands in L1 and the NoC command can race ahead of it: a blocking load of the
    // last word drains the stores first (same trick as masked_bincount).
    (void)ckernel::load_blocking(running_sum + (W - 1));
    noc.async_write(out0, dst_totals_accessor, totals_page_size, {}, {.page_id = 0});
    noc.async_write_barrier();

    // 2. Compute exclusive prefix sum of tile-aligned totals, grouped by experts_per_chip
    //    Pad each expert's count to TILE_HEIGHT so each expert starts at a tile boundary
    //    For group [a, b, c, d] -> [0, align(a), align(a)+align(b), align(a)+align(b)+align(c)]
    // One group per chip model-wide, which is not H: with several dispatch groups there are more
    // expert groups than devices on the cluster axis.
    constexpr uint32_t num_expert_groups = W / experts_per_chip;
    for (uint32_t g = 0; g < num_expert_groups; g++) {
        uint32_t prefix = 0;
        for (uint32_t i = 0; i < experts_per_chip; i++) {
            uint32_t idx = g * experts_per_chip + i;
            uint32_t val = running_sum[idx];
            running_sum[idx] = prefix;
            prefix += (val + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT * tt::constants::TILE_HEIGHT;
        }
    }

    // 3. Write expert region offsets (shared component, before adding any local offset) to DRAM.
    //    running_sum is only read from here on, so this write shares the final barrier.
    (void)ckernel::load_blocking(running_sum + (W - 1));
    noc.async_write(out0, dst_expert_region_accessor, expert_region_page_size, {}, {.page_id = 0});

    // 4. Row r is expert_region + (rows 0..r-1). `offsets` is written from row row_idx of the same
    //    buffer, so the two outputs cannot disagree.
    constexpr uint32_t row_stride = all_offsets_page_size / sizeof(uint32_t);
    uint32_t* rows = reinterpret_cast<uint32_t*>(cpu_l1_addr(rows_scratch));
    for (uint32_t i0 = 0; i0 < W_blocked; i0 += COLS) {
        emit_rows<COLS, H, in_stride, row_stride>(in, running_sum, rows, i0);
    }
    for (uint32_t i0 = W_blocked; i0 < W; i0++) {
        emit_rows<1, H, in_stride, row_stride>(in, running_sum, rows, i0);
    }
    (void)ckernel::load_blocking(rows + (H - 1) * row_stride + (W - 1));
    for (uint32_t r = 0; r < H; r++) {
        noc.async_write(
            rows_scratch,
            dst_all_offsets_accessor,
            all_offsets_page_size,
            {.offset_bytes = r * all_offsets_page_size},
            {.page_id = r});
    }
    noc.async_write(
        rows_scratch,
        dst_offsets_accessor,
        offsets_page_size,
        {.offset_bytes = row_idx * all_offsets_page_size},
        {.page_id = 0});
    noc.async_write_barrier();
}

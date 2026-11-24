// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_accessor.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/csa_compressor/device/kernels/csa_compressor_kernel_utils.hpp"

namespace {

using namespace csa_compressor::kernel_utils;

// Scratch slots. One source tile of kv and gate at a time, plus the predecessor state pair that
// window 0 reads. The position bias goes straight into its own DFB for the compute kernel.
constexpr uint32_t kScratchKv = 0;
constexpr uint32_t kScratchGate = 1;
constexpr uint32_t kScratchPredecessorKv = 2;
constexpr uint32_t kScratchPredecessorScore = 3;

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

}  // namespace

void kernel_main() {
    const uint32_t output_tiles = get_arg(args::output_tiles);
    const uint32_t complete_windows = get_arg(args::complete_windows);
    const uint32_t absolute_start = get_arg(args::absolute_start);
    const uint32_t first_output_tile = get_arg(args::first_output_tile);

    constexpr uint32_t input_width_tiles = get_arg(args::input_width_tiles);
    constexpr uint32_t output_width_tiles = get_arg(args::output_width_tiles);

    const auto kv = TensorAccessor(tensor::kv);
    const auto gate = TensorAccessor(tensor::gate);
    const auto bias = TensorAccessor(tensor::bias);
    const auto predecessor_kv = TensorAccessor(tensor::predecessor_kv);
    const auto predecessor_score = TensorAccessor(tensor::predecessor_score);

    DataflowBuffer candidate_kv(dfb::candidate_kv);
    DataflowBuffer candidate_score(dfb::candidate_score);
    // ca_bias / cb_bias: the position bias of the Ca and Cb halves of the projection.
    DataflowBuffer ca_bias(dfb::ca_bias);
    DataflowBuffer cb_bias(dfb::cb_bias);
    // The scratch slots above, filled and drained by this kernel alone.
    Scratchpad<uint16_t> scratch(scratch::scratch);
    volatile tt_l1_ptr uint16_t* source = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cpu_l1_addr(scratch));
    Noc noc;

    // Candidate tile c, row r holds candidate c of window (32 * output_tile_row + r), which turns the
    // softmax over the 8 candidates into elementwise tile math downstream. Building that by walking
    // (window, candidate) re-reads the same source tile up to 256 times to keep 32 values out of each,
    // so this walks source tiles instead and scatters every row of one to the candidates it feeds.
    //
    // Source tile row j, local row i, holds token 32j + i. As a Cb candidate it feeds its own window
    // (8j + i/4); as a Ca candidate it feeds the NEXT one. Both halves therefore land at candidate
    // (half * 4 + i % 4) and row (window - 32 * output_tile_row) of that candidate's tile, and j only
    // has to sweep the four (Ca: five) source tile rows whose windows fall inside this output tile.
    const uint32_t output_tile_end = first_output_tile + output_tiles;
    for (uint32_t output_tile = first_output_tile; output_tile < output_tile_end; ++output_tile) {
        candidate_kv.reserve_back(8);
        candidate_score.reserve_back(8);
        ca_bias.reserve_back(1);
        cb_bias.reserve_back(1);
        volatile tt_l1_ptr uint16_t* candidate_kv_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(candidate_kv.get_write_ptr());
        volatile tt_l1_ptr uint16_t* candidate_score_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(candidate_score.get_write_ptr());
        for (uint32_t i = 0; i < 8 * kTileElems; ++i) {
            candidate_kv_ptr[i] = 0;
            candidate_score_ptr[i] = 0;
        }

        const uint32_t output_tile_row = output_tile / output_width_tiles;
        const uint32_t feature_tile = output_tile % output_width_tiles;
        const uint32_t window_begin = 32 * output_tile_row;

        // Every row of a candidate tile carries the same slot, so its whole bias is one row of the
        // bias tensor and the compute kernel can broadcast it. Ca reads the left half, Cb the right.
        noc.async_read(bias, ca_bias, kTileBytes, {.page_id = feature_tile}, {.offset_bytes = 0});
        noc.async_read(bias, cb_bias, kTileBytes, {.page_id = output_width_tiles + feature_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();

        // Windows past the chunk's real length stay zero, which is the pooled value they must produce.
        if (window_begin < complete_windows) {
            const uint32_t source_row_begin = 4 * output_tile_row;

            for (uint32_t half = 0; half < 2; ++half) {
                // Ca is the left half of the projection and feeds the window after its own.
                const uint32_t candidate_base = 4 * half;
                const uint32_t window_offset = half == 0 ? 1 : 0;
                const uint32_t column_tile = half * output_width_tiles + feature_tile;
                const uint32_t first_source_row =
                    (half == 0 && output_tile_row > 0) ? source_row_begin - 1 : source_row_begin;

                for (uint32_t j = first_source_row; j <= source_row_begin + 3; ++j) {
                    // Windows grow with j, so once this tile row starts past the end so does every
                    // later one. Bounding the read this way also keeps it inside the padded slab.
                    if (8 * j + window_offset >= complete_windows) {
                        break;
                    }
                    const uint32_t page = j * input_width_tiles + column_tile;
                    noc.async_read(
                        kv, scratch, kTileBytes, {.page_id = page}, {.offset_bytes = kScratchKv * kTileBytes});
                    noc.async_read(
                        gate, scratch, kTileBytes, {.page_id = page}, {.offset_bytes = kScratchGate * kTileBytes});
                    noc.async_read_barrier();
                    invalidate_l1_cache();

                    for (uint32_t i = 0; i < 32; ++i) {
                        const uint32_t window = 8 * j + i / 4 + window_offset;
                        if (window < window_begin || window >= window_begin + 32 || window >= complete_windows) {
                            continue;
                        }
                        const uint32_t candidate = candidate_base + (i & 3);
                        const uint32_t output_row = window - window_begin;
                        copy_tile_row(
                            candidate_kv_ptr + candidate * kTileElems, source + kScratchKv * kTileElems, output_row, i);
                        copy_tile_row(
                            candidate_score_ptr + candidate * kTileElems,
                            source + kScratchGate * kTileElems,
                            output_row,
                            i);
                    }
                }
            }

            // Window 0 has no predecessor window on this chip, so its Ca candidates come from the
            // state the exchange handed us. Those scores already carry the position bias, and the
            // compute kernel is about to broadcast it over the whole tile, so take it back out here.
            // Only these four rows pay for the round trip.
            if (output_tile_row == 0) {
                const uint32_t parity = (absolute_start / 4) & 1;
                const uint32_t page = parity * output_width_tiles + feature_tile;
                noc.async_read(
                    predecessor_kv,
                    scratch,
                    kTileBytes,
                    {.page_id = page},
                    {.offset_bytes = kScratchPredecessorKv * kTileBytes});
                noc.async_read(
                    predecessor_score,
                    scratch,
                    kTileBytes,
                    {.page_id = page},
                    {.offset_bytes = kScratchPredecessorScore * kTileBytes});
                noc.async_read_barrier();
                invalidate_l1_cache();

                volatile tt_l1_ptr uint16_t* ca_bias_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ca_bias.get_write_ptr());
                for (uint32_t slot = 0; slot < 4; ++slot) {
                    copy_tile_row(
                        candidate_kv_ptr + slot * kTileElems, source + kScratchPredecessorKv * kTileElems, 0, slot);
                    for (uint32_t col = 0; col < 32; ++col) {
                        // The state row and the bias row are both `slot`, so one offset serves both.
                        const uint32_t offset = tile_offset(slot, col);
                        const float unbiased = bf16_to_float(source[kScratchPredecessorScore * kTileElems + offset]) -
                                               bf16_to_float(ca_bias_ptr[offset]);
                        candidate_score_ptr[slot * kTileElems + tile_offset(0, col)] = float_to_bf16_rne(unbiased);
                    }
                }
            }
        }

        candidate_kv.push_back(8);
        candidate_score.push_back(8);
        ca_bias.push_back(1);
        cb_bias.push_back(1);
    }
}

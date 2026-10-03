// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "kernel_math.hpp"
#include "kernels/attention/compute/attention_util.hpp"
#include "kernels/misc/print.hpp"
#include "kernels/data_format_convert.hpp"
#include "kernels/general/common.hpp"
#include "kernels/dataflow/end_queue_signal_writer.hpp"
#include "kernels/attention/compute/softmax_util.hpp"

#include <kernel_api/common.hpp>
#include <kernel_api/aloc.hpp>
#include <kernel_api/dataflow/api.h>
#include <kernel_api/tiling_utils.hpp>
#include <kernel_api/untilize.hh>
#include <kernel_api/local_buffer.hpp>
#include <kernel_api/inter_tile_matmul.hpp>
#include <kernel_api/fp_math.hpp>
#include <kernel_api/binary_reductions.hpp>
#include <kernel_api/block_wise_softmax.hpp>
#include <kernel_api/tile_util.hpp>

using namespace tt::tt_metal;

namespace {

enum softmax_mode {
    SCALE,
    LOGSUMEXP,
};

static inline void apply_softmax_scale(
    local_tensor<float>& in, local_tensor<float>& out, float scale) {
    constexpr auto TILE_H = tile_h<float>();
    constexpr auto TILE_W = tile_w<float>();
    for (int r = 0; r < TILE_H; r++) {
        for (int c = 0; c < TILE_W; c++) {
            out[r][c] = in[r][c] * scale;
        }
    }
}

static inline void fill_with_minus_inf(local_tensor<float>& t) {
    constexpr auto TILE_H = tile_h<float>();
    constexpr auto TILE_W = tile_w<float>();
    for (int r = 0; r < TILE_H; r++) {
        for (int c = 0; c < TILE_W; c++) {
            t[r][c] = -INFINITY;
        }
    }
}

// Helper to check if a row is fully masked
static inline bool is_row_fully_masked(const uint8_t* mask_tile, int tile_w, int row, int col_from, int col_to) {
    int count = 0;
    for (int c = col_from; c < col_to; c++) {
        if (mask_tile[row * tile_w + c] == 0) {
            count++;
        }
    }
    return count == (col_to - col_from);
}

// Helper to compute softmax denominator with masked values
static inline float softmax_denom(
    local_tensor<float>& logits,
    const uint8_t* mask_tile,
    int valid_cols,
    int tile_w,
    int row,
    int mask_valid_cols,
    bool use_causal_mask) {
    float max_val = -INFINITY;
    float sum = 0.0f;

    // Find max over valid columns
    for (int c = 0; c < valid_cols; c++) {
        max_val = fmaxf(max_val, logits[row][c]);
    }
    // Also consider masked columns that might have higher values but should be -inf
    // If there's padding (valid_cols < TILE_W), padded columns contribute -inf

    for (int c = 0; c < valid_cols; c++) {
        float val = logits[row][c];
        if (mask_tile != nullptr) {
            // Apply mask: if masked, value becomes -inf
            bool masked = (mask_tile[row * tile_w + c] == 0);
            if (masked) {
                val = -INFINITY;
            }
        }
        sum += expf(val - max_val);
    }

    // Padded columns contribute -inf (i.e., 0 to the sum)
    return max_val, sum;
}

static inline std::pair<float, float> compute_softmax_stats(
    local_tensor<float>& logits,
    const uint8_t* mask_tile,
    int valid_cols,
    int tile_w,
    int row,
    bool use_causal_mask) {
    float max_val = -INFINITY;
    float sum = 0.0f;

    // Find max over valid columns
    for (int c = 0; c < valid_cols; c++) {
        max_val = fmaxf(max_val, logits[row][c]);
    }

    for (int c = 0; c < valid_cols; c++) {
        float val = logits[row][c];
        if (mask_tile != nullptr) {
            // Apply mask: if masked, value becomes -inf
            bool masked = (mask_tile[row * tile_w + c] == 0);
            if (masked) {
                val = -INFINITY;
            }
        }
        sum += expf(val - max_val);
    }

    return {max_val, sum};
}

static inline void apply_softmax(
    local_tensor<float>& logits,
    local_tensor<float>& out,
    const uint8_t* mask_tile,
    int valid_cols,
    int tile_w,
    bool use_causal_mask) {
    constexpr auto TILE_H = tile_h<float>();

    for (int r = 0; r < TILE_H; r++) {
        auto [max_val, sum] = compute_softmax_stats(
            logits, mask_tile, valid_cols, tile_w, r, use_causal_mask);

        // If sum is 0 (all -inf), set all outputs to 0
        if (sum == 0.0f || sum != sum) {  // sum is NaN or 0
            for (int c = 0; c < valid_cols; c++) {
                out[r][c] = 0.0f;
            }
        } else {
            for (int c = 0; c < valid_cols; c++) {
                float val = logits[r][c];
                if (mask_tile != nullptr) {
                    bool masked = (mask_tile[r * tile_w + c] == 0);
                    if (masked) {
                        out[r][c] = 0.0f;
                    } else {
                        out[r][c] = expf(val - max_val) / sum;
                    }
                } else {
                    out[r][c] = expf(val - max_val) / sum;
                }
            }
        }
    }
}

}  // namespace

// ============================================================================
// Softmax kernel for 2D tensors (standard path)
// ============================================================================
void softmax_2d(
    uint32_t input_l1_address,
    uint32_t output_l1_address,
    uint32_t scale,
    uint32_t reduction_axes_bitmask,
    uint32_t keep_dims) {
    const auto in_addr = reinterpret_cast<float*>(input_l1_address);
    const auto out_addr = reinterpret_cast<float*>(output_l1_address);

    constexpr auto TILE_H = tile_h<float>();
    constexpr auto TILE_W = tile_w<float>();
    constexpr uint32_t NUM_TILES = get_noc_multicore_dest_args_dest_idx(get_write_tile_params());

    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        noc_async_read_tile(0, in_addr, t);
        noc_async_read_tile(1, in_addr, t);
        noc_async_read_barrier();

        local_tensor<float> in_0 = noc_async_read_tile_get_local_buffer<0>(t);
        local_tensor<float> in_1 = noc_async_read_tile_get_local_buffer<1>(t);

        // Determine which axis to reduce over
        bool reduce_rows = (reduction_axes_bitmask & 0x1) != 0;
        bool reduce_cols = (reduction_axes_bitmask & 0x2) != 0;

        if (reduce_rows && !reduce_cols) {
            // Reduce over rows - compute softmax across tiles in the row
            local_tensor<float> out = noc_async_read_tile_get_local_buffer<0>(t);
            for (int r = 0; r < TILE_H; r++) {
                for (int c = 0; c < TILE_W; c++) {
                    out[r][c] = in_0[r][c];
                }
            }
            // TODO: implement row reduction
        } else if (!reduce_rows && reduce_cols) {
            // Reduce over columns - apply softmax within each tile
            local_tensor<float> out = noc_async_read_tile_get_local_buffer<0>(t);
            apply_softmax(in_0, out, nullptr, TILE_W, TILE_W, false);
            noc_async_write_tile(0, out_addr, t);
        } else {
            // Default: apply softmax over columns
            local_tensor<float> out = noc_async_read_tile_get_local_buffer<0>(t);
            apply_softmax(in_0, out, nullptr, TILE_W, TILE_W, false);
            noc_async_write_tile(0, out_addr, t);
        }

        noc_async_write_barrier();
    }
}

// ============================================================================
// Softmax kernel for attention (with masking support)
// ============================================================================
void softmax_attention(
    uint32_t logits_l1_address,
    uint32_t output_l1_address,
    uint32_t mask_l1_address,
    uint32_t scale,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t seq_len,
    uint32_t use_causal_mask,
    uint32_t mask_pad_padded_data) {
    const auto logits_addr = reinterpret_cast<float*>(logits_l1_address);
    const auto output_addr = reinterpret_cast<float*>(output_l1_address);
    const auto mask_addr = (mask_l1_address != 0) ? reinterpret_cast<uint8_t*>(mask_l1_address) : nullptr;

    constexpr auto TILE_H = tile_h<float>();
    constexpr auto TILE_W = tile_w<float>();
    constexpr uint32_t NUM_TILES = get_noc_multicore_dest_args_dest_idx(get_write_tile_params());

    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        noc_async_read_tile(0, logits_addr, t);
        if (mask_addr != nullptr) {
            noc_async_read_tile(1, mask_addr, t);
        }
        noc_async_read_barrier();

        local_tensor<float> logits = noc_async_read_tile_get_local_buffer<0>(t);
        local_tensor<uint8_t> mask = (mask_addr != nullptr) ?
            noc_async_read_tile_get_local_buffer<1>(t) : local_tensor<uint8_t>();

        local_tensor<float> out = noc_async_read_tile_get_local_buffer<0>(t);

        // Copy logits to output initially
        for (int r = 0; r < TILE_H; r++) {
            for (int c = 0; c < TILE_W; c++) {
                out[r][c] = logits[r][c];
            }
        }

        // Apply scale
        if (scale != 1.0f) {
            apply_softmax_scale(out, out, scale);
        }

        int valid_cols = TILE_W;
        bool use_mask = (mask_addr != nullptr) || (mask_pad_padded_data != 0);

        // Handle tile padding: if mask_pad_padded_data is set, apply -inf to padded columns
        if (mask_pad_padded_data != 0) {
            // Compute how many columns are padding based on seq_len
            int tile_idx = t % (seq_len / TILE_W + (seq_len % TILE_W != 0 ? 1 : 0));
            int offset_in_seq = tile_idx * TILE_W;
            int remaining = seq_len - offset_in_seq;
            valid_cols = (remaining < TILE_W) ? remaining : TILE_W;

            // Apply -inf to padded columns
            for (int c = valid_cols; c < TILE_W; c++) {
                for (int r = 0; r < TILE_H; r++) {
                    out[r][c] = -INFINITY;
                }
            }
        }

        // Apply mask if present
        if (mask_addr != nullptr && mask.data() != nullptr) {
            for (int r = 0; r < TILE_H; r++) {
                for (int c = 0; c < valid_cols; c++) {
                    bool masked = (mask[r * TILE_W + c] == 0);
                    if (masked) {
                        out[r][c] = -INFINITY;
                    }
                }
            }
        }

        // Apply softmax
        apply_softmax(out, out, (mask_addr != nullptr && mask.data() != nullptr) ?
            reinterpret_cast<uint8_t*>(mask.data()) : nullptr, valid_cols, TILE_W, use_causal_mask != 0);

        noc_async_write_tile(0, output_addr, t);
        noc_async_write_barrier();
    }
}

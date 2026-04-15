// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

/**
 * @file matmul_helpers_compute.hpp
 * @brief Unified matmul compute helper with absolute-offset packing
 *
 * Single entry point for both standard matmul and SDPA. Uses absolute-offset
 * packing (pack_tile<true>) so subblock dims can be chosen purely for compute
 * efficiency, independent of output layout. Output tiles land in row-major
 * order in the CB.
 *
 * Standard matmul: num_k_blocks > 1, K-blocking with spill/reload
 * SDPA: num_k_blocks = 1, single-pass, all K tiles available upfront
 */

namespace compute_kernel_lib {

// =============================================================================
// MatmulMode — selects tile vs block LLK functions
// =============================================================================

enum class MatmulMode { TILE, BLOCK };

inline constexpr MatmulMode TILE = MatmulMode::TILE;
inline constexpr MatmulMode BLOCK = MatmulMode::BLOCK;

// =============================================================================
// MatmulConfig — runtime configuration
// =============================================================================

struct MatmulConfig {
    uint32_t in0_cb_id;
    uint32_t in1_cb_id;
    uint32_t out_cb_id;

    uint32_t ct_dim = 1;  // subblock_w
    uint32_t rt_dim = 1;  // subblock_h
    uint32_t kt_dim = 1;  // in0_block_w (K tiles per block)

    bool transpose = false;
    uint32_t partials_cb_id = 0;  // intermediate CB for spill/reload

    static constexpr MatmulConfig block(
        uint32_t in0,
        uint32_t in1,
        uint32_t out,
        uint32_t ct,
        uint32_t rt,
        uint32_t kt,
        bool trans = false,
        uint32_t partials = 0) {
        return {in0, in1, out, ct, rt, kt, trans, partials};
    }
};

// =============================================================================
// Functor defaults
// =============================================================================

struct NoPostCompute {
    ALWI void operator()(uint32_t) const {}
};

struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// =============================================================================
// detail:: — Internal building blocks (no DST management)
// =============================================================================

namespace detail {

template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx);

template <MatmulMode mode>
ALWI void matmul_accumulate(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t dst_start,
    uint32_t count,
    uint32_t in0_stride,
    uint32_t in1_stride,
    uint32_t dst_stride);

}  // namespace detail

// =============================================================================
// Initialization
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg);

// =============================================================================
// Unified matmul helper
// =============================================================================

/**
 * @brief Blocked matmul with K-blocking and absolute-offset packing.
 *
 * Handles both standard matmul (num_k_blocks > 1) and SDPA (num_k_blocks = 1).
 * Output tiles are packed at row-major positions via pack_tile<true>.
 *
 * K-blocking (num_k_blocks > 1):
 *   - Non-last K-blocks: spill partial results to partials_cb (sequential pack)
 *   - Last K-block: pack output at absolute offsets to out_cb (or partials_cb
 *     if pack_last_to_interm), push per M-subblock row-group
 *
 * Single-pass (num_k_blocks = 1):
 *   - No spill/reload
 *   - Pack output at absolute offsets, push per row-group
 *
 * PREREQUISITE: Caller must call mm_block_init() and reconfig_data_format()
 * before invoking this helper.
 *
 * @tparam packer_l1_acc      Enable packer L1 accumulation (avoids spill/reload)
 * @tparam pack_last_to_interm Pack last K-block to partials_cb instead of out_cb
 * @tparam pack_relu           Enable PACK_RELU on last K-block (when !pack_last_to_interm)
 * @param retain_in0           When true, skip popping in0 on the last K-block so the
 *                             caller retains the data (e.g. SDPA reuses Q across K chunks)
 */
template <
    MatmulMode mode,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    bool pack_relu = false,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock>
ALWI void matmul_blocks_absolute(
    const MatmulConfig& cfg,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t num_k_blocks,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    bool retain_in0 = false);

/**
 * @brief Inplace reduce via matmul, used by SDPA matmul_reduce.
 */
template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(
    const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles, uint32_t total_in0_tiles);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl"

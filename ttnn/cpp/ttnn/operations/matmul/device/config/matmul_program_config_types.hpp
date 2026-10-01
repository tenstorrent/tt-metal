// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul {

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1
// for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool transpose_mcast{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    bool fuse_batch = true;
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
    // Fused SwiGLU epilogue (opt-in). in1 holds tile-pair interleaved [gate | up] columns (weight tile 2p = gate
    // tile p, tile 2p+1 = up tile p, as from prepare_for_fused_swiglu). The output is silu(gate) * up, so its
    // width is half the weight width. Last member so the struct stays a positional aggregate.
    bool fuse_swiglu = false;
    // Fused SwiGLU variants (opt-in, need fuse_swiglu). glu_last_block: apply silu(gate) * up on DEST inside the
    // last K block (the plain reload path) and pack half the tiles, instead of a separate pass over the partials.
    // glu_sfpu_on_pack: issue the SwiGLU SFPU work from the PACK thread (with glu_last_block it then overlaps the
    // MATH thread's next subblock). Env TT_MATMUL_GLU_SFPU_ON_PACK=1 also sets it.
    bool glu_last_block = false;
    bool glu_sfpu_on_pack = false;
    // in0 CB holds one K block instead of two (saves out_block_h * in0_block_w in0 tiles of L1 per core; the in0
    // multicast of block k+1 then waits for compute to release block k). Opt-in.
    bool in0_single_buffer = false;
    // Two in1 senders per column (opt-in). Every in1 K block is split in two halves of in0_block_w / 2 K rows. The
    // top-row core of each column reads and multicasts half A over NOC_0 (as the single sender does today), the
    // bottom-row core reads half B and multicasts it to the rows above it (also NOC_0, wrapping around the NoC
    // torus); every core of the column waits for both halves before it computes on the block. Same blocks, same K
    // order, same CB layout: bit-exact vs a single sender. Needs the interleaved no-bias path, !transpose_mcast, an
    // even in0_block_w and at least 3 core rows with no M padding. Last member so the struct stays a positional
    // aggregate.
    bool in1_dual_sender = false;
};

// 1D mcast matmul program config.
//
// When `gather_in0 == false`, `compute_with_storage_grid_size` describes the size of the
// rectangular grid of worker cores that the multicast paths will use, anchored at (0, 0) on
// the device, or at the bounding-box start of the active sub-device when `sub_device_id` is
// set on the op. The 1D mcast factory targets a single bounding-box rectangle for multicast
// and the per-core index math assumes a single contiguous row-major rectangle, so when
// `sub_device_id` is provided the sub-device's worker cores must themselves form a single
// rectangle. Non-rectangular sub-device grids are rejected at validate time.
//
// When `gather_in0 == true`, `compute_with_storage_grid_size` is ignored and the gather path
// can run on any sub-device worker layout, including non-rectangular ones.
struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool fuse_batch{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    bool mcast_in0{};
    bool gather_in0{};
    CoreRangeSet hop_cores;
    std::size_t num_global_cb_receivers{};
    bool untilize_out{};
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
    // Select ring-rotated FIFO delivery for gather_in0 with a DRAM-sender GCB. The feeding prefetcher
    // request MUST supply a per-receiver rotation. GCB-backed mcast_in0 instead consumes natural FIFO
    // order and requires this flag to remain false.
    bool stream_in1 = false;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    std::size_t num_workers_per_dram_bank = 1;
};

struct MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig {
    std::size_t in0_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>;

// True when the config is the 2D multicast config with the fused SwiGLU epilogue enabled.
inline bool is_fuse_swiglu(const std::optional<MatmulProgramConfig>& config) {
    if (!config.has_value()) {
        return false;
    }
    const auto* mcast2d = std::get_if<MatmulMultiCoreReuseMultiCastProgramConfig>(&config.value());
    return mcast2d != nullptr && mcast2d->fuse_swiglu;
}

// Ensures allowed_worker_cores is populated on every config variant that supports it.
// If allowed_worker_cores is already set, it is left unchanged.  Otherwise it is
// synthesized from compute_with_storage_grid_size (or from the device grid for
// MatmulMultiCoreProgramConfig).  After this call, factories can read
// config.allowed_worker_cores.value() unconditionally.
inline void normalize_program_config(MatmulProgramConfig& config, const tt::tt_metal::CoreCoord& device_grid) {
    auto make_crs = [](const tt::tt_metal::CoreCoord& grid) {
        return CoreRangeSet(CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
    };
    std::visit(
        [&](auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (
                std::is_same_v<T, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<T, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (!c.allowed_worker_cores.has_value()) {
                    c.allowed_worker_cores = make_crs(c.compute_with_storage_grid_size);
                }
            } else if constexpr (std::is_same_v<T, MatmulMultiCoreProgramConfig>) {
                if (!c.allowed_worker_cores.has_value()) {
                    c.allowed_worker_cores = make_crs(device_grid);
                }
            }
            // DRAM-sharded configs have no grid fields to normalize.
        },
        config);
}

}  // namespace ttnn::operations::matmul

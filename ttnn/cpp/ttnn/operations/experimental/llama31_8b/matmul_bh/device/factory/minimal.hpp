// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"

namespace ttnn::experimental::prim::llama31_8b_matmul_bh {

inline constexpr uint32_t kIn1SegmentSlots = 24;

struct MinimalPlanDefine {
    const char* key;
    const char* value;
};

struct MinimalPlanRegion {
    const char* name;
    uint8_t x0, y0, x1, y1;
    uint8_t in0_sender, in1_sender;
};

struct MinimalPlanCore {
    uint8_t x, y, in0_idx, in1_idx;
    uint8_t in0_sender, in0_sink, in0_prev_x, in0_prev_y, in0_next_x, in0_next_y;
    uint8_t in1_sender, in1_sink, in1_prev_x, in1_prev_y, in1_next_x, in1_next_y;
    uint16_t N_start_tile, N_end_tile, defer_write_k_block;
};

struct MinimalPlan {
    uint32_t grid_x, grid_y;
    bool transpose;
    uint32_t in0_cores, in1_cores, in0_noc, in1_noc;
    uint32_t K_tiles, padded_K_tiles, N_tiles, padded_N_tiles;
    uint32_t M_block_tiles, K_block_tiles, N_block_tiles;
    uint32_t N_blocks_per_core, K_blocks, subblock_h, subblock_w;
    uint32_t in0_cb_tiles, in1_cb_tiles, out_cb_tiles, interm_cb_tiles;
    uint32_t max_defer_write_k_block, bulk_width;
    bool in1_wmc;
    uint32_t swiglu_block_float_output;
    uint32_t chunk_tiles[4];
    uint32_t num_heads, num_kv_heads, head_dim, heads_dt, heads_qt, heads_kt;
    const MinimalPlanDefine* defines;
    uint32_t num_defines;
    const MinimalPlanRegion* regions;
    uint32_t num_regions;
    const MinimalPlanCore* cores;
    uint32_t num_cores;
    const uint8_t* bulk_counts;
    const uint32_t (*bulk_segments)[kIn1SegmentSlots];
};

struct HeadsLayout {
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
};

ttnn::device_operation::ProgramArtifacts create_minimal_program_artifacts(
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const std::optional<HeadsLayout>& heads);

struct MinimalProgramFactory {
    using operation_attributes_t = MinimalMatmulParams;
    using tensor_args_t = MinimalMatmulInputs;
    using tensor_return_value_t = std::vector<Tensor>;

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MinimalMatmulParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const MinimalMatmulParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim::llama31_8b_matmul_bh

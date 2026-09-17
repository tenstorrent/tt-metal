// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Everything the op's validation, its output-spec derivation and the factory need from a
// MatmulUnifiedProgramConfig plus the operand shapes. One function computes it so the three can never
// disagree about which core owns which piece of C or how big the buffers are. Every constraint of the
// config is checked in that function with TT_FATAL, so calling it is the config check.
//
// Vocabulary (classic GEMM, all sizes in 32x32 tiles): C[M x N] = A[M x K] x B[K x N].
//   block        the per_core_M x per_core_N tiles of C a core produces in one go; cores walk C in blocks
//                row-major (across N, then down M), batch after batch
//   subblock     the subblock_M_tiles x subblock_N_tiles tiles of a block accumulated in DST at once
//   K iteration  K_iteration_tiles of the inner dimension; one A slice + one B slice per iteration
struct UnifiedMatmulPlan {
    uint32_t M_tiles = 0;
    uint32_t K_tiles = 0;
    uint32_t N_tiles = 0;
    uint32_t batch_size = 0;             // batches of A (and of C)
    bool broadcast_B_over_batch = true;  // one B for every batch, or a B per batch

    // Blocking, after the config's auto fields are resolved.
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    uint32_t K_iteration_tiles = 0;
    uint32_t num_K_iterations = 0;  // K_tiles / K_iteration_tiles
    uint32_t subblock_M_tiles = 0;
    uint32_t subblock_N_tiles = 0;

    // Block assignment. C is walked in blocks row-major (across N, then down M), batch after batch;
    // active core i starts at (first_batch[i], first_M_tile[i], first_N_tile[i]) and produces
    // num_blocks[i] consecutive blocks of that walk.
    uint32_t total_blocks = 0;  // over all batches
    bool row_major_cores = true;
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> first_batch;
    std::vector<uint32_t> first_M_tile;
    std::vector<uint32_t> first_N_tile;
    std::vector<uint32_t> num_blocks;
    uint32_t max_blocks_per_core = 0;

    // Dataflow-buffer rings. A slot holds one tile; slot sizes are in bytes.
    bool packer_l1_acc_en = false;
    tt::DataFormat A_format{};
    tt::DataFormat B_format{};
    tt::DataFormat C_format{};
    tt::DataFormat C_partials_format{};
    uint32_t A_slot_bytes = 0;
    uint32_t B_slot_bytes = 0;
    uint32_t C_slot_bytes = 0;
    uint32_t C_partials_slot_bytes = 0;
    uint32_t A_slice_ring_slots = 0;
    uint32_t B_slice_ring_slots = 0;
    uint32_t C_block_ring_slots = 0;
    uint32_t C_partials_ring_slots = 0;
    // C_partials shares C_block's L1; only safe when partials are never live while C_block holds unread data.
    bool alias_C_partials_onto_C_block = false;
    uint64_t l1_bytes = 0;  // total ring footprint per core

    // Only valid for a sharded output: the shard layout implied by how the blocks tile C.
    tt::tt_metal::TensorMemoryLayout sharded_output_layout() const;
};

UnifiedMatmulPlan plan_unified_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes);

struct MatmulUnifiedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::qsr

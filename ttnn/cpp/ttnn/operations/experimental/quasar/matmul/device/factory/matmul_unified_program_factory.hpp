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
// Vocabulary (classic GEMM, all sizes in 32x32 tiles): C[M x N] = A[M x K] x B[K x N], per batch.
//   C slice     the C_slice_M_tiles x C_slice_N_tiles tiles of C a core produces in one go: the
//                L1-fittable piece of the core's output region. Normally the region is one C slice; a large
//                region is produced as several consecutive C slices (across N, then down M). Every core
//                produces its C slices for every batch
//   subblock     the subblock_M_tiles x subblock_N_tiles tiles of a C slice accumulated in DST at once (one
//                matmul_block call per K tile); "block" means this and nothing else
//   K chunk  K_chunk_tiles of the inner dimension; one A slice + one B slice per K chunk
struct UnifiedMatmulPlan {
    uint32_t M_tiles = 0;
    uint32_t K_tiles = 0;
    uint32_t N_tiles = 0;
    uint32_t batch_size = 0;             // batches of A (and of C)
    bool broadcast_B_over_batch = true;  // one B for every batch, or a B per batch

    // Blocking, after the config's auto fields are resolved.
    uint32_t C_slice_M_tiles = 0;
    uint32_t C_slice_N_tiles = 0;
    uint32_t K_chunk_tiles = 0;
    uint32_t num_K_chunks = 0;  // K_tiles / K_chunk_tiles
    uint32_t subblock_M_tiles = 0;
    uint32_t subblock_N_tiles = 0;

    // C slice assignment. The C slices of one batch are walked row-major (across N, then down M) and the
    // walk is split into contiguous runs, one per active core; core i starts at
    // (first_C_slice_M_tile[i], first_C_slice_N_tile[i]) and produces num_C_slices[i] C slices, for every
    // batch.
    uint32_t C_slices_per_batch = 0;
    bool row_major_cores = true;
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> first_C_slice_M_tile;
    std::vector<uint32_t> first_C_slice_N_tile;
    std::vector<uint32_t> num_C_slices;
    uint32_t max_C_slices_per_core = 0;

    // Borrowing: an L1-sharded operand whose shard on every active core is exactly what that core's rings
    // would hold is bound as the ring itself (DFB borrowed_from), so nothing is copied. A: the shard is the
    // chunk's rows for all of K (one K chunk, chunks span N). B: the shard is the chunk's columns for all of
    // K (chunks span M; K chunks are contiguous runs of it). C: the finished C slice is packed straight into
    // the shard, which needs subblock-major pack order to equal the shard's row-major tile order, i.e.
    // subblock_N_tiles == C_slice_N_tiles; the writer then only waits. All three need one C slice per core
    // and batch 1, and a shard grid that lists the active cores in assignment order. Borrowed rings cost
    // no extra L1.
    bool borrow_A = false;
    bool borrow_B = false;
    bool borrow_C = false;

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
    uint32_t C_slice_ring_slots = 0;
    uint32_t C_partials_ring_slots = 0;
    // C_partials shares C_slice's L1; only safe when partials are never live while C_slice holds unread data.
    bool alias_C_partials_onto_C_slice = false;
    uint64_t l1_bytes = 0;  // total ring footprint per core

    // Only valid for a sharded output: the shard layout implied by how the C slices tile C.
    tt::tt_metal::TensorMemoryLayout sharded_output_layout() const;
};

// `output` is the C tensor when the caller supplied one (or the op already created it); it decides whether
// C can be packed in place. Without it the op allocates C from the plan, which matches by construction.
UnifiedMatmulPlan plan_unified_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes,
    const std::optional<ttnn::Tensor>& output);

struct MatmulUnifiedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::qsr

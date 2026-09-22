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

// Derived from a MatmulUnifiedProgramConfig + operand shapes by one function shared by validation,
// output-spec derivation and the factory, so they cannot disagree (all config constraints TT_FATAL there).
struct UnifiedMatmulPlan {
    uint32_t M_tiles = 0;
    uint32_t K_tiles = 0;
    uint32_t N_tiles = 0;
    uint32_t batch_size = 0;  // batches of A (and of C)
    // Tiles from one batch of B to the next: K_tiles * N_tiles when B has a [K x N] per batch, 0 when B is a
    // single [K x N] (or [1 x K x N]) that every batch of A multiplies. A's stride is always M_tiles * K_tiles.
    uint32_t B_batch_stride_tiles = 0;

    // Blocking, after the config's auto fields are resolved.
    uint32_t C_slice_M_tiles = 0;
    uint32_t C_slice_N_tiles = 0;
    uint32_t K_chunk_tiles = 0;
    uint32_t num_K_chunks = 0;  // K_tiles / K_chunk_tiles
    uint32_t subblock_M_tiles = 0;
    uint32_t subblock_N_tiles = 0;
    // C slice dims rounded up to subblock multiples (equal when the subblock divides the slice).
    // Buffers and kernel loops use these; overshoot rows/columns are clipped on write.
    uint32_t C_slice_M_padded_tiles = 0;
    uint32_t C_slice_N_padded_tiles = 0;

    // C slice assignment: one batch's C slices, walked across N then down M, split into contiguous
    // runs per active core (the factory derives the per-core RTAs).
    uint32_t C_slices_per_batch = 0;
    bool row_major_cores = true;
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t max_C_slices_per_core = 0;  // sizes the DFBs and gates partials aliasing

    // Borrowed operand: its L1 shard on each active core is bound as the DFB itself, no copy.
    // Needs batch 1, one C slice per core, and a shard grid in assignment order.
    bool borrow_A = false;
    bool borrow_B = false;
    bool borrow_C = false;

    // DFB sizing. An entry holds one tile; entry sizes are in bytes.
    bool packer_l1_acc_en = false;
    tt::DataFormat A_format{};
    tt::DataFormat B_format{};
    tt::DataFormat C_format{};
    tt::DataFormat C_partials_format{};
    uint32_t A_entry_bytes = 0;
    uint32_t B_entry_bytes = 0;
    uint32_t C_entry_bytes = 0;
    uint32_t C_partials_entry_bytes = 0;
    uint32_t A_slice_entries = 0;
    uint32_t B_slice_entries = 0;
    uint32_t C_slice_entries = 0;
    uint32_t C_partials_entries = 0;
    // C_partials shares C_slice's L1; only safe when partials are never live while C_slice holds unread data.
    bool alias_C_partials_onto_C_slice = false;
    uint64_t l1_bytes = 0;  // total DFB footprint per core

    // Only meaningful for a sharded output: the layout implied by how the C slices tile C.
    tt::tt_metal::TensorMemoryLayout sharded_output_layout{};
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

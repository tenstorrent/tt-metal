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
// disagree about which core owns which output block or how big the buffers are. Every constraint of
// the config is checked in that function with TT_FATAL, so calling it is the config check.
struct UnifiedMatmulPlan {
    // Problem size, in tiles. B is the in0 batch; in1 is either broadcast (bcast_batch) or batched alike.
    uint32_t B = 0;
    uint32_t Mt = 0;
    uint32_t Kt = 0;
    uint32_t Nt = 0;
    bool bcast_batch = true;

    // Blocking, after auto fields are resolved.
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    uint32_t in0_block_w = 0;
    uint32_t out_subblock_h = 0;
    uint32_t out_subblock_w = 0;
    uint32_t num_blocks_inner_dim = 0;  // Kt / in0_block_w

    // Output block grid: block i is at (i / num_block_cols, i % num_block_cols).
    uint32_t num_block_rows = 0;
    uint32_t num_block_cols = 0;
    uint32_t num_blocks = 0;

    // Active cores in assignment order; core i owns blocks [block_start[i], block_start[i] + blocks_per_core[i]).
    bool row_major_cores = true;
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> block_start;
    std::vector<uint32_t> blocks_per_core;
    uint32_t max_blocks_per_core = 0;

    // Dataflow buffers (entry sizes in bytes, capacities in entries).
    bool packer_l1_acc_en = false;
    tt::DataFormat in0_format{};
    tt::DataFormat in1_format{};
    tt::DataFormat out_format{};
    tt::DataFormat interm_format{};
    uint32_t in0_entry_size = 0;
    uint32_t in1_entry_size = 0;
    uint32_t out_entry_size = 0;
    uint32_t interm_entry_size = 0;
    uint32_t in0_entries = 0;
    uint32_t in1_entries = 0;
    uint32_t out_entries = 0;
    uint32_t interm_entries = 0;
    bool alias_out_interm = false;  // interm shares the out ring (safe only when interm is never live across blocks)
    uint64_t l1_bytes = 0;          // total DFB footprint per core

    // Only valid for a sharded output: the shard layout implied by the block grid.
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

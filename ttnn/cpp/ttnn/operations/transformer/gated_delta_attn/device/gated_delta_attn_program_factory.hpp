// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_device_operation_types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct GatedDeltaAttnSeqSharedVars {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    uint32_t grid_y = 1;     // rows per column — used for column-major (head,v-slice)→core mapping
    uint32_t num_cores = 0;  // = num_heads * split_v
    uint32_t split_v = 1;    // value-dim cores per head
    uint32_t Vt_global = 0;  // full value-tile count (for slice addressing)
};

struct GatedDeltaAttnSeqProgramFactory {
    using shared_variables_t = GatedDeltaAttnSeqSharedVars;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GatedDeltaAttnSeqParams& op_attrs,
        const GatedDeltaAttnSeqInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GatedDeltaAttnSeqParams& op_attrs,
        const GatedDeltaAttnSeqInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    uint32_t grid_y = 1;  // rows per column — used for column-major head→core mapping
    uint32_t num_cores = 0;
};

struct GatedDeltaAttnSeqProgramFactory {
    using shared_variables_t = GatedDeltaAttnSeqSharedVars;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GatedDeltaAttnSeqParams& attrs, const GatedDeltaAttnSeqInputs& in, std::vector<Tensor>& outputs);

    static void override_runtime_arguments(
        cached_program_t& cached,
        const GatedDeltaAttnSeqParams& attrs,
        const GatedDeltaAttnSeqInputs& in,
        std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim

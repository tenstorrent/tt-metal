// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "extract_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::experimental::quasar::deepseek_prefill::extract {

struct ExtractProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ExtractParams& operation_attributes, const ExtractInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::quasar::deepseek_prefill::extract

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "insert_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::experimental::quasar::deepseek_prefill::insert {

struct InsertProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const InsertParams& operation_attributes, const InsertInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::quasar::deepseek_prefill::insert

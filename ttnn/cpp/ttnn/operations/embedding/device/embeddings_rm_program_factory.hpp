// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct EmbeddingsRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

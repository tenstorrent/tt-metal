// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_backward_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct EmbeddingBackwardProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const EmbeddingBackwardParams& operation_attributes,
        const EmbeddingBackwardInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

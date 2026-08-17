// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::prim {

struct UntilizeCodegenOperationAttributes;
struct UntilizeCodegenTensorArgs;

struct UntilizeCodegenProgramFactory {
    // Builds the codegen untilize program for the (already validated, already output-allocated)
    // case. This is also the op's ONLY live-L1 decision point: it samples the free L1 headroom
    // once and, if no codegen CB plan fits below the resident L1 buffers, emits the native
    // untilize program for the same tensors instead of failing. Runs only on a program-cache
    // miss; cache hits patch the cached program's buffer addresses without re-entering here.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeCodegenOperationAttributes& operation_attributes,
        const UntilizeCodegenTensorArgs& tensor_args,
        const Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

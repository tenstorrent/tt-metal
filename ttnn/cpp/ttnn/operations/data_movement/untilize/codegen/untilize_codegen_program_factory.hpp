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
    // Builds the codegen untilize program (or native-equivalent) for the already-validated,
    // already output-allocated case. Live L1 is sampled on every dispatch in
    // compute_program_hash (choose_codegen_cb_plan) so a CB-tier / Native-block-split change
    // is a cache miss. create_descriptor itself still runs only on a miss.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeCodegenOperationAttributes& operation_attributes,
        const UntilizeCodegenTensorArgs& tensor_args,
        const Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

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
    // Builds the codegen untilize program for the already-validated, already output-allocated
    // case. Live L1 is sampled on every dispatch in compute_program_hash (choose_codegen_cb_plan)
    // so a CB-tier change is a cache miss. create_descriptor itself still runs only on a miss.
    // It never builds anything but a codegen program: a case for which no codegen CB plan fits
    // the free L1 is routed to the native prim by ttnn::untilize before dispatch, and is a hard
    // error if it reaches this factory anyway.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeCodegenOperationAttributes& operation_attributes,
        const UntilizeCodegenTensorArgs& tensor_args,
        const Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

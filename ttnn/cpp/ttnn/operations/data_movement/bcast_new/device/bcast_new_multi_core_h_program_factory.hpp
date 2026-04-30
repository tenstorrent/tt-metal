// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bcast/device/bcast_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Shared variables for MULTI_CORE_H strategy — superseded by ProgramDescriptor (Phase 1 _new path).

struct BcastNewMultiCoreHProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmReaderWriterMultiCoreDefaultProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim

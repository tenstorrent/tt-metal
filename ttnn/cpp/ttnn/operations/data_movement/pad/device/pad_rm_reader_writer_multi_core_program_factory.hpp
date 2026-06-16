// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmReaderWriterMultiCoreProgramFactory {
    // The pad-value const tensor is allocated once on a cache miss inside create_program_spec()
    // and returned in ProgramArtifacts::op_owned_tensors; the framework parks it at a stable
    // address for the cached Program's life and refreshes only the io tensor bindings on a hit.
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim

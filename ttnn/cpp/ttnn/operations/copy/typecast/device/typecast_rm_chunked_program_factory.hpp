// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "typecast_device_op_types.hpp"

namespace ttnn::prim {

struct TypecastRowMajorChunkedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim

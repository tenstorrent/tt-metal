// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim::qsr {

struct TypecastRowMajorChunkedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr

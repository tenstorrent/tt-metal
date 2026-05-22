// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "convert_to_chw_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct ConvertToCHWProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  Both input and output CBs are
    // sharded (set_globally_allocated_address bound to their respective tensor
    // buffers); the framework re-applies CB addresses on cache-hit via
    // apply_descriptor_runtime_args.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConvertToCHWParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim

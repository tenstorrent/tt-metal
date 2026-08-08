// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::for_python {

// Metal 1.0 copy of the production factory, frozen for the Python fusion framework.
// Not part of MatmulDeviceOperation and not on any dispatch path — do not port it to Metal 2.0.

struct MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
};

}  // namespace ttnn::for_python

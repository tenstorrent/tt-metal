// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/matmul_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim::qsr {

struct MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::qsr::MatmulParams& operation_attributes,
        const ttnn::prim::qsr::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
};

}  // namespace ttnn::prim::qsr

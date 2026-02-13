// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::matmul_new_detail {

struct DRAMShardedDescriptorFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::matmul_new_detail

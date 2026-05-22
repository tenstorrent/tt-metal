// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct SparseMatmulMultiCoreReuseMcast1DProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct SparseMatmulMultiCoreReuseMcast1DProgramFactory {
    // Metal 2.0 factory: create_program_artifacts returns a ProgramSpec plus the ProgramRunArgs
    // that populate it. Every value that varies between two dispatches sharing a program hash is a
    // tensor base address -- in0, in1, sparsity (or indices), and output -- and each reaches its
    // kernel through a typed TensorParameter/TensorBinding, which the framework refreshes in place
    // on a cache hit. Everything else is derived from the hashed tensor specs and operation
    // attributes, so a hit guarantees it is already correct.
    //
    // The shape of this struct is what routes the op onto that path: ProgramSpecFactoryConcept
    // matches on a lone create_program_artifacts. There is no override_runtime_arguments, so the
    // framework -- not this factory -- owns the cache-hit refresh.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim

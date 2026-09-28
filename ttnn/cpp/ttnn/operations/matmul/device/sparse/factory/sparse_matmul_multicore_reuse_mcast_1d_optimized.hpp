// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct SparseMatmulMultiCoreReuseMcast1DProgramFactory {
    // Every value that varies between two dispatches sharing a program hash is a buffer address:
    // in0, in1, sparsity (or indices), and output. create_descriptor declares each as a Buffer*
    // binding via KernelDescriptor::emplace_runtime_args, so the framework patches it in place on a
    // cache hit. Everything else is derived from the hashed tensor specs and operation attributes,
    // so a hit guarantees it is already correct.
    //
    // The shape of this struct is what routes the op onto that path: ProgramDescriptorFactoryConcept
    // matches on a lone create_descriptor, and a cached_program_t or cached_mesh_workload_t typedef
    // would select the CachedProgram path instead.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"

namespace ttnn::prim {

// Sparse matmul with in1 delivered over the Tensor prefetcher's PrefetcherPipes (mask mode): the dense
// Metal 2.0 mcast_in0 body over pipes plus its SPARSITY paths. Selected when prefetcher_pipes is set;
// the ProgramDescriptor factory keeps every other sparse configuration.
//
// A lone create_program_artifacts is what ProgramSpecFactoryConcept matches: everything that differs
// between two dispatches sharing a program hash (in0, in1, sparsity and output addresses, and the pipe
// objects) is a run argument, so the framework rebinds it on a cache hit.
struct SparseMatmulMultiCoreReuseMcast1DSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SparseMatmulParams& operation_attributes,
        const SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim

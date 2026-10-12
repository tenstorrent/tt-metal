// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "dram_prefetcher_device_operation_types.hpp"

namespace ttnn::prim {

// Delivers into worker-sender PrefetcherPipes (`prefetcher_pipes`). A Program binds a pipe only through a
// ProgramSpec, so this path is a Metal 2.0 spec factory beside the GlobalCircularBuffer descriptor factory.
//
// A lone create_program_artifacts is what ProgramSpecFactoryConcept matches: the address tensor is the
// one run argument that differs between dispatches sharing a program hash (the pipes are part of the
// hash), so the framework rebinds it on a cache hit.
struct DramPrefetcherPipeSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

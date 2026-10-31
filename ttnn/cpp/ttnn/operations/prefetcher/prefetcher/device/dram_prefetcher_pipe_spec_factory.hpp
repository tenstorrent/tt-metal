// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>

#include "dram_prefetcher_device_operation_types.hpp"

namespace ttnn::prim {

// Delivers into worker-sender PrefetcherPipes (`prefetcher_pipes`). A Program binds a pipe only through a
// ProgramSpec, so this path is a Metal 2.0 spec factory beside the GlobalCircularBuffer descriptor factory.
//
// A ProgramSpecFactoryConcept factory: on a cache hit the framework rebinds the address tensor and the
// pipes. The pipes key the hash by geometry and addresses, so a hit may bring pipes re-carved in place
// of the ones the program was built with.
struct DramPrefetcherPipeSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& tensor_return_value);

    static tt::tt_metal::experimental::
        Table<tt::tt_metal::experimental::PrefetcherPipeParamName, tt::tt_metal::experimental::PrefetcherPipeArgument>
        prefetcher_pipe_args(const DramPrefetcherParams& operation_attributes);
};

}  // namespace ttnn::prim

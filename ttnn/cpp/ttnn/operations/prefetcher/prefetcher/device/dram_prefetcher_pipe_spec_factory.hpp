// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/command_list_params.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include "dram_prefetcher_device_operation_types.hpp"

namespace ttnn::prim {

// Delivers into worker-sender PrefetcherPipes (`prefetcher_pipes`). A Program binds a pipe only through a
// ProgramSpec, so this path is a Metal 2.0 spec factory beside the GlobalCircularBuffer descriptor factory.
//
// A lone create_program_artifacts is what ProgramSpecFactoryConcept matches: the address tensor and the
// pipes are the run arguments that differ between dispatches sharing a program hash (a pipe keys the
// hash by its geometry and addresses, so a hit may bring pipes re-carved in place of the program's), and
// the framework rebinds both on a cache hit.
struct DramPrefetcherPipeSpecFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& tensor_return_value);

    // Binds keyword prefetcher_pipes, so a command list can move the prefetcher to other pipes.
    static ttnn::command_list::CmdListBindings map_cmd_list_parameters(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        const tt::tt_metal::ProgramSpec& spec,
        const tt::tt_metal::ProgramRunArgs& run_params);
};

}  // namespace ttnn::prim

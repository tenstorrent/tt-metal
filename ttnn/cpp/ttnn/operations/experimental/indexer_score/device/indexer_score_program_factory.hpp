// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "indexer_score_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::indexer_score::program {

struct IndexerScoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel;
    tt::tt_metal::KernelHandle compute_kernel;
    tt::tt_metal::KernelHandle writer_kernel;
    std::vector<CoreCoord> worker_cores;
};

struct IndexerScoreProgramFactory {
    using shared_variables_t = IndexerScoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& args, const tensor_args_t& tensors, tensor_return_value_t& out);

    static void override_runtime_arguments(
        cached_program_t& cached,
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out);
};

}  // namespace ttnn::operations::experimental::indexer_score::program

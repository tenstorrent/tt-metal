// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::indexer_score::program

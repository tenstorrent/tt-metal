// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek::indexer::program {

// Skeleton: empty program, no kernels. The output tensor is allocated but
// never written, so it holds garbage — enough to prove the op compiles,
// dispatches and returns the right shape (and fails PCC, as expected).
IndexerScoreProgramFactory::cached_program_t IndexerScoreProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    return {std::move(program), IndexerScoreSharedVariables{}};
}

void IndexerScoreProgramFactory::override_runtime_arguments(
    cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}

}  // namespace ttnn::operations::experimental::deepseek::indexer::program

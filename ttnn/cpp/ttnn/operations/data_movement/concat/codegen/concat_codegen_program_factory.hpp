// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Default read_batch==write_batch for the non-width RM builders
// (build_concat_rm / build_concat_rm_nonwidth_nway).
inline constexpr uint32_t kConcatNonWidthBatch = 4;
// Default write_batch for the width RM builders (build_concat_rm_width /
// build_concat_rm_width_nway); their reader is unbatched (2-tensor) or reads
// with a separate, unscaled read_batch=1 (N-way), so only write_batch (and the
// CB depth it drives) is scaled for L1 fit.
inline constexpr uint32_t kConcatWidthWriteBatch = 4;
// ops/concat/spec.py's MAX_RM_NONWIDTH_NWAY_INPUTS: the nonwidth N-way reader's
// runtime-arg count is 3 + 3*N words, comfortably under the portable
// runtime-argument bound at N=64. spec.py defines no separate bound for
// build_concat_rm_width_nway (RT layout 2 + 3*N, the same order), so this
// ceiling is reused for both N-way builders.
inline constexpr uint32_t kConcatMaxNwayInputs = 64;

// Largest batch <= max_batch for which a depth=2*batch CB of `page_size`
// bytes/page fits `l1_budget_bytes`; nullopt if even batch=1 (depth=2) does
// not fit. Single source shared by the routing gate
// (concat_codegen_supported.cpp) and the factory, so they cannot drift.
std::optional<uint32_t> plan_concat_batch(uint32_t page_size, uint32_t max_batch, uint64_t l1_budget_bytes);

// Per-core L1 budget available to a statically allocated CB, mirroring
// concat_program_factory.cpp's ConcatProgramFactory `l1_capacity` computation.
uint32_t concat_l1_budget(tt::tt_metal::IDevice* device);

struct ConcatCodegenParams {
    uint32_t dim{};
    uint32_t num_inputs{};
    uint32_t stick_size{};
    uint32_t total_out_sticks{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ConcatCodegenInputs {
    std::vector<Tensor> input_tensors;
};

struct ConcatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatCodegenParams& operation_attributes,
        const ConcatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

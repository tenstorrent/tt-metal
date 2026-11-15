// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::sample_candidates {

struct SampleCandidatesParams {
    uint32_t k = 32;
    float fill_value = -1.0e9F;
    uint32_t num_workers = 64;
    MemoryConfig output_mem_config;
};

struct SampleCandidatesInputs {
    const Tensor& logits;
    const Tensor& inv_temp;
    const Tensor& index_offset;
};

using SampleCandidatesSpecReturn = std::vector<tt::tt_metal::TensorSpec>;
using SampleCandidatesTensorReturn = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::deepseek::sample_candidates

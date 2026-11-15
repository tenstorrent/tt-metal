// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "sample_candidates_device_operation_types.hpp"
#include "sample_candidates_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::sample_candidates {

struct SampleCandidatesDeviceOperation {
    using operation_attributes_t = SampleCandidatesParams;
    using tensor_args_t = SampleCandidatesInputs;
    using spec_return_value_t = SampleCandidatesSpecReturn;
    using tensor_return_value_t = SampleCandidatesTensorReturn;
    using program_factory_t = std::variant<SampleCandidatesProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::sample_candidates

namespace ttnn::prim {

std::vector<Tensor> decode_sample_candidates(
    const Tensor& logits,
    const Tensor& inv_temp,
    const Tensor& index_offset,
    uint32_t k,
    float fill_value,
    uint32_t num_workers);

}  // namespace ttnn::prim

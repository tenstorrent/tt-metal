// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::transformer {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_concat_heads_decode(
    const Tensor& input_tensor, Tensor& output, CoreCoord compute_with_storage_grid_size);

tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_concat_heads_decode_subcoregrids(
    const Tensor& input_tensor, Tensor& output, CoreCoord compute_with_storage_grid_size);

struct NLPConcatHeadsDecodeDeviceOperation {
    const uint32_t num_heads;
    const bool on_subcoregrids = false;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::transformer

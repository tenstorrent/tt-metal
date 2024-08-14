// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::transformer {

operation::ProgramWithCallbacks multi_core_nlp_kv_cache_load_slice(const Tensor &a, Tensor& output, const tt::tt_metal::Shape &output_tensor_start, const tt::tt_metal::Shape &output_tensor_end);

struct NlpKVCacheLoadSliceDeviceOperation {
    const tt::tt_metal::Shape output_tensor_start;
    const tt::tt_metal::Shape output_tensor_end;
    const tt::tt_metal::Shape output_shape;
    const tt::tt_metal::Shape input_shape;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::transformer

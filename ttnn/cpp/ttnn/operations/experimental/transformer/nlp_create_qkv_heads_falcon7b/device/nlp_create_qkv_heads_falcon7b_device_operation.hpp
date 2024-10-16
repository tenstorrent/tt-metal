// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/run_operation.hpp"
#include <variant>

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_falcon7b(const Tensor& input_tensor_a,
                                                                         std::vector<Tensor>& output,
                                                                         CoreCoord compute_with_storage_grid_size);

struct NlpCreateHeadsFalcon7BDeviceOperation {
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors,
                                                   std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::transformer

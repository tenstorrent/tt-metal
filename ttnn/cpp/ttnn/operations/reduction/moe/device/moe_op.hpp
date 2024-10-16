// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction {

struct MoeDeviceOperation {
    const uint16_t k;
    const MemoryConfig output_mem_config;

    void validate_with_output_tensors(const std::vector<Tensor>& input_tensors,
                                      const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors,
                                                   std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::reduction

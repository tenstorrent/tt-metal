// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

// Device operation following TTNN operation pattern
struct UpSample3DDeviceOperation {
    const uint32_t scale_factor_d_;
    const uint32_t scale_factor_h_;
    const uint32_t scale_factor_w_;
    const tt::tt_metal::MemoryConfig output_mem_config_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::upsample

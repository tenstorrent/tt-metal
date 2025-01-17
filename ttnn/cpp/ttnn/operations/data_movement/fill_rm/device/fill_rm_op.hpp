// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

// Generates an NCHW row-major tensor and fill it with ones up to hOnes, wOnes in each HW tile
// with the rest padded with zeros. So for H=2, W=3, hFill=1, wFill=2 the following tensor will be generated:
// +------------> W
// | hi hi lo
// | lo lo lo
// |
// v H
//
// H, W are expected to be multiples of 32
// The 'any' Tensor arg is only used to pass the device and resulting tensor dtype
// val_hi/lo are expected to be uint16 encodings of bfloat16 numbers, so 0x3f80 for 1.0 etc.
struct FillRM {
    uint32_t N, C, H, W, hFill, wFill;
    float val_hi, val_lo;
    const tt::tt_metal::MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::data_movement

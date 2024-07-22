// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace tt {

namespace tt_metal {

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
struct FillRM  {
    uint32_t N, C, H, W, hFill, wFill;
    float val_hi, val_lo;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

Tensor fill_rm (uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const Tensor& any, float val_hi, float val_lo, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

inline
Tensor fill_ones_rm (uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hOnes, uint32_t wOnes, const Tensor& any, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    // 0x3f80 is 1.0 in BF16
    return fill_rm(N, C, H, W, hOnes, wOnes, any, 1.0, 0.0, output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt

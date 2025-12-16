// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"
namespace ttnn::operations::op_slicing {

struct Op2DSliceConfig {
    // Determines the dimension along which the input & output tensors are sliced.
    // Slices based on [N, H, W, C] shape.
    // Using width slicing is more efficient as it reduces memory usage. This is because the overlap of data between
    // cores is minimized in width slicing, reducing the size of the Halo output. If the Height & Width dimensions are
    // similar, then use Width slicing. Use Height slicing if the Height dimension is significantly larger than the
    // Width dimension.
    enum class SliceType : uint8_t {
        DRAM_HEIGHT,
        DRAM_WIDTH,
        L1_FULL  // This option can be used to force conv2d with a DRAM Input to move it to L1, and output will be in
                 // L1.
    };
    SliceType slice_type = SliceType::DRAM_WIDTH;

    // Number of slices that the output tensor should be divided into.
    uint32_t num_slices = 0;
};

class OpSliceAttr {
public:
    virtual ~OpSliceAttr() = default;
    using IOShape = std::tuple<uint32_t, uint32_t>;
    virtual std::tuple<IOShape, IOShape> get_input_slice(IOShape output_slice_start, IOShape output_slice_end) = 0;

    virtual uint32_t get_L1_usage() = 0;
    virtual tt::tt_metal::MemoryConfig get_input_memory_config(
        IOShape output_slice_start, IOShape output_slice_end) = 0;
    virtual ttnn::Tensor run_L1_op(
        const ttnn::Tensor& sliced_input_tensor, IOShape output_slice_start, IOShape output_slice_end) = 0;
    virtual std::string name() = 0;
};
void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& output_tensor,
    OpSliceAttr* op_slice_attr,
    Op2DSliceConfig dram_slice_config);

}  // namespace ttnn::operations::op_slicing

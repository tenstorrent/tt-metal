// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "conv2d/device/conv2d_op.hpp"

namespace ttnn::operations::slicing_ops {

class OpSliceAttr {
public:
    virtual ~OpSliceAttr() = default;

    using IOShape = std::tuple<uint32_t, uint32_t>;
    virtual std::tuple<IOShape, IOShape> get_input_slice(IOShape output_slice_start, IOShape output_slice_end);

    virtual uint32_t get_L1_usage();
    virtual tt::tt_metal::MemoryConfig get_input_memory_config(IOShape output_slice_start, IOShape output_slice_end);
    virtual ttnn::Tensor run_L1_op(const ttnn::Tensor& sliced_tensor);
};
void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& output_tensor,
    OpSliceAttr* op_slice_attr,
    conv::conv2d::Conv2dSliceConfig dram_slice_config);
}  // namespace ttnn::operations::slicing_ops

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/shape/shape.hpp"
namespace ttnn::operations::slicing_ops {

class IOSlicing {
    virtual std::tuple<ttnn::Shape, ttnn::Shape> get_input_slice(
        ttnn::Shape output_slice_start, ttnn::Shape output_slice_end);
virtual uint32_t get_L1_usage() public : virtual ~IOSlicing() = default;
};
}  // namespace ttnn::operations::slicing_ops

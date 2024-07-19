// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/experimental/tt_dnn/op_library/composite/composite_ops.hpp"

namespace tt {
namespace tt_metal {
    // vector returns param, exp_avg, exp_avg_sq
    std::vector<Tensor> lamb_optimizer(const Tensor& data, const Tensor& grad, const Tensor& exp_avg, const Tensor& exp_avg_sq, float beta1, float beta2, float step_size, float eps, float weight_decay, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
}
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce {

struct operation_attributes_t {
    uint32_t num_links = 2;
    uint32_t ring_size = 2;
    tt::tt_fabric::Topology topology{};
    std::optional<uint32_t> cluster_axis;
};

struct tensor_args_t {
    Tensor input_tensor;
    std::optional<Tensor> intermediate_tensor;
    std::optional<Tensor> residual_tensor;  // Optional residual input for fused residual add
    std::optional<Tensor> persistent_output_tensor;  // Optional persistent output buffer
};

}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce

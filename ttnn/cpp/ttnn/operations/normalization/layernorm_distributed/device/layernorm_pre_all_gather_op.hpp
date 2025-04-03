// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "layernorm_distributed_types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace normalization {
enum class LayerNormDistributedType;
}  // namespace normalization
}  // namespace operations
}  // namespace ttnn
namespace tt {
namespace tt_metal {
enum class DataType;
}  // namespace tt_metal
}  // namespace tt

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

tt::tt_metal::operation::ProgramWithCallbacks layernorm_pre_allgather_multi_core(
    const Tensor& a,
    Tensor& output,
    LayerNormDistributedType norm_type,
    DeviceComputeKernelConfig compute_kernel_config);

struct LayerNormPreAllGather {
    LayerNormDistributedType norm_type;
    const DataType dtype;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::normalization

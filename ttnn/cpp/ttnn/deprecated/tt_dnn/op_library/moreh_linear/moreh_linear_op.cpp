// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_linear/moreh_linear_op.hpp"

#include <type_traits>

#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

inline void moreh_linear_validate(
    const Tensor& weight) {
    const auto& weight_shape = weight.get_legacy_shape();
    const auto& weight_rank = weight_shape.rank();
    TT_FATAL(weight_rank == 2, "weight rank {} must be 2.", weight_rank);
}

Tensor _moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<const Tensor>& bias,
    std::optional<Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    moreh_linear_validate(weight);
    output = moreh_matmul(input, weight, false, true, output, bias, output_mem_config);
    return output.value();
}

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    std::optional<const Tensor> bias,
    std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return _moreh_linear(input, weight, bias, output, output_mem_config, compute_kernel_config);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

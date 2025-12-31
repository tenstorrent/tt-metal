// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling.hpp"
#include "device/sampling_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

Tensor sampling(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& preallocated_output_tensor) {
    using OperationType = operations::reduction::sampling::SamplingDeviceOperation;
    return device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.seed = seed, .sub_core_grids = sub_core_grids},
        OperationType::tensor_args_t{
            .input_values = input_values_tensor,
            .input_indices = input_indices_tensor,
            .k = k,
            .p = p,
            .temp = temp,
            .preallocated_output = preallocated_output_tensor});
}

}  // namespace ttnn

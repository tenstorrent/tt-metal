
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/rmsnorm_fw_device_operation.hpp"
#include "rmsnorm_fw.hpp"

namespace ttnn::operations::experimental {

std::vector<std::optional<Tensor>> RMSNormForwardOperation::invoke(
    const Tensor& input_tensor, const Tensor& gamma_tensor, bool return_intermediates, float epsilon) {
    auto result = ttnn::prim::rmsnorm_fw(input_tensor, gamma_tensor, return_intermediates, epsilon);

    if (result.size() == 1U) {
        return {result[0], std::nullopt};
    }

    return {result[0], result[1]};
}

std::vector<std::optional<Tensor>> RMSNormForwardOperation::create_async_optional_output_tensors(
    const Tensor& input_tensor, const Tensor& gamma_tensor, bool return_intermediates, float epsilon) {
    return {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})),
        return_intermediates
            ? std::optional<Tensor>(Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor})))
            : std::nullopt};
}

}  // namespace ttnn::operations::experimental

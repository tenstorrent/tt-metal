
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/rmsnorm_fw_device_operation.hpp"
#include "rmsnorm_fw.hpp"

namespace ttnn::operations::experimental {

// std::vector<Tensor> RMSNormForwardOperation::invoke(
std::tuple<Tensor, Tensor> RMSNormForwardOperation::invoke(
    const Tensor& input_tensor, const Tensor& gamma_tensor, bool return_intermediates, float epsilon) {
    // return ttnn::prim::rmsnorm_fw(
    //     input_tensor, gamma_tensor, return_intermediates, epsilon);

    // DEBUG
    auto res = ttnn::prim::rmsnorm_fw(input_tensor, gamma_tensor, return_intermediates, epsilon);
    return {res[0], res[1]};
}

}  // namespace ttnn::operations::experimental

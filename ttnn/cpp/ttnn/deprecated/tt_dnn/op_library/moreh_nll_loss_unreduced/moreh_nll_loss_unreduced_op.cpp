// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_unreduced/moreh_nll_loss_unreduced_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {


Tensor moreh_nll_loss_unreduced(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const MemoryConfig& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    const Tensor& result = moreh_nll_loss_step2(
        input_tensor,
        target_tensor,
        weight_tensor,
        std::nullopt,
        ignore_index,
        "sum",
        memory_config,
        compute_kernel_config);

    return result;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt

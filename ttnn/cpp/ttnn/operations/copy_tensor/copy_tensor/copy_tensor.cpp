// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_tensor.hpp"
#include <optional>

#include "device/copy_tensor_op.hpp"

namespace ttnn::operations::copy_tensor {

Tensor ExecuteCopyTensor::invoke(const Tensor& src_tensor, const Tensor& dst_tensor) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({src_tensor}))};
    operation::launch_op(
        [src_tensor, dst_tensor](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(CopyTensor{}, {src_tensor, dst_tensor});
        },
        {src_tensor, dst_tensor},
        output_tensors);

    return output_tensors.at(0);
}

}  // namespace ttnn::operations::copy_tensor

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "downsample.hpp"
#include "device/downsample_op.hpp"

namespace ttnn::operations::downsample {

Tensor ExecuteDownsample::invoke(const Tensor& input_tensor_a,
                                 std::array<uint32_t, 5> downsample_params,
                                 std::optional<DataType> dtype) {
    auto dtype_ = dtype.has_value() ? dtype.value() : input_tensor_a.get_dtype();
    auto output_tensor = operation::run(Downsample{downsample_params, dtype_}, {input_tensor_a}).front();
    return output_tensor;
};

}  // namespace ttnn::operations::downsample

// SPDX-License-Identifier: Apache-2.0
#include "my_matmul.hpp"

namespace ttnn {

Tensor my_matmul(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    return ttnn::prim::my_matmul(input_tensor_a, input_tensor_b);
}

}  // namespace ttnn

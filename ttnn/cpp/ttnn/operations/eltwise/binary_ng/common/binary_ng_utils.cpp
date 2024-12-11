// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/binary_ng/common/binary_ng_utils.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::binary_ng::utils {

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_dtype,
    const std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    std::string idst = "tile_id";
    std::map<std::string, std::string> defines;

    if (fused_activations.has_value()) {
        defines.merge(ttnn::operations::unary::utils::get_block_defines(fused_activations.value(), "0", idst));
    }
    return defines;
}
}  // namespace ttnn::operations::binary_ng::utils

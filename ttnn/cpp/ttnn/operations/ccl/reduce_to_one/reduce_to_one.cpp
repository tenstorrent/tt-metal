// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/reduce_to_one_op.hpp"
#include "reduce_to_one.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteReduceToOne::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& root_coord,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    return ttnn::prim::reduce_to_one(input_tensor, topology, root_coord, optional_output_tensor);
}

}  // namespace ttnn::operations::ccl

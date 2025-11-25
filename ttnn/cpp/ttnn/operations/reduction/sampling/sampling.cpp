// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling.hpp"

#include "device/sampling_device_operation.hpp"

namespace ttnn::operations::reduction::sampling {

Tensor ExecuteSampling::invoke(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k_tensor,
    const Tensor& p_tensor,
    const Tensor& temp_tensor,
    const std::optional<uint32_t>& seed,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::sampling(
        input_values_tensor,
        input_indices_tensor,
        k_tensor,
        p_tensor,
        temp_tensor,
        seed,
        sub_core_grids,
        output_tensor);
}

}  // namespace ttnn::operations::reduction::sampling

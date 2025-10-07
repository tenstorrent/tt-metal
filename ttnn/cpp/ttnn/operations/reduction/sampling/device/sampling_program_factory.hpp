// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

tt::tt_metal::operation::ProgramWithCallbacks sampling_multicore_interleaved(
    const std::vector<Tensor>& input_tensors,
    const std::optional<uint32_t>& seed,
    const std::optional<CoreRangeSet>& sub_core_grids,
    Tensor& output_tensor);

}  // namespace ttnn::operations::reduction::detail

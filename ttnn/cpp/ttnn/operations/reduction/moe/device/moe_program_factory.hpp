// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {


operation::ProgramWithCallbacks moe_single_core_interleaved(const Tensor &input_tensor, const Tensor &expert_mask_tensor, const Tensor &topk_mask_tensor, const uint16_t k, Tensor &out_tensor);


}

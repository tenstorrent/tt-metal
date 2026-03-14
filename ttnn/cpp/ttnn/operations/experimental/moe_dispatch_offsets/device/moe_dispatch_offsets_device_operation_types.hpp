// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeDispatchOffsetsParams {
    const uint32_t n_routed_experts;
};

}  // namespace ttnn::experimental::prim

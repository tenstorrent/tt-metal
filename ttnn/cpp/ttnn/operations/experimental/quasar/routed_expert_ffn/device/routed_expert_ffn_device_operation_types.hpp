// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

struct RoutedExpertFfnParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RoutedExpertFfnInputs {
    Tensor x;
    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
};

}  // namespace ttnn::prim::qsr

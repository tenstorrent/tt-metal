// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

struct SimpleAddParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct SimpleAddInputs {
    Tensor input_a;
    Tensor input_b;
};

}  // namespace ttnn::prim::qsr

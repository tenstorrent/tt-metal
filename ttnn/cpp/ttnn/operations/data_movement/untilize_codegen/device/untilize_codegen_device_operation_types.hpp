// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct UntilizeCodegenParams {
    // Output memory config is the only free parameter; dtype/shape/Wt/Ht/core-split
    // and the L1-aware CB depths are all derived from the input tensor spec (already
    // in the cache key) and recomputed in create_descriptor.
    tt::tt_metal::MemoryConfig m_output_mem_config;
};

struct UntilizeCodegenInputs {
    Tensor input;
};

}  // namespace ttnn::prim

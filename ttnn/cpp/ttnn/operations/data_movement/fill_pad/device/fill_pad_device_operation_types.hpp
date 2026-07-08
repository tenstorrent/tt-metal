// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct FillPadParams {
    tt::tt_metal::PadValue fill_value;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct FillPadInputs {
    Tensor input;
};

}  // namespace ttnn::prim

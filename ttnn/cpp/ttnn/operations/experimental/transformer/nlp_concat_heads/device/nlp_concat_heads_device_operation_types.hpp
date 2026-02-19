// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::experimental::prim {

struct NlpConcatHeadsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

}  // namespace ttnn::experimental::prim

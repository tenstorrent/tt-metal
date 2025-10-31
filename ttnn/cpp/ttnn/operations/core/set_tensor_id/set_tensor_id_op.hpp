// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

struct SetTensorId {
    static std::uint64_t invoke(std::uint64_t id);
};

}  // namespace ttnn::operations::core

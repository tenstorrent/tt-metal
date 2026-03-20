// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include <variant>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace ttnn::operations::unary {

Tensor frac(const Tensor&, const std::optional<MemoryConfig>&);
Tensor is_odd(const Tensor&, const std::optional<MemoryConfig>&);

}  // namespace ttnn::operations::unary

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/bfloat16.hpp>

#include "ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn {

namespace operations::ternary {
Tensor _addcmul(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _addcdiv(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _lerp(
    const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&, const std::optional<Tensor>&);
Tensor _lerp_overload(
    const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&, const std::optional<Tensor>&);

}  // namespace operations::ternary

Tensor mac(const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor mac(const Tensor&, float, float, const std::optional<MemoryConfig>&);

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace ttnn::operations::unary {

enum class UnaryCompositeOpType {
    DIGAMMA,
    LGAMMA,
    MULTIGAMMALN,
    VAR_HW,
    STD_HW,
    NORMALIZE_HW,
    GLU,
    REGLU,
    GEGLU,
    SWIGLU,
    POW,
    TRIL,
    TRIU,
    POLYGAMMA,
    LOGICAL_NOT_,
    NORMALIZE_GLOBAL,
    FRAC,
};
Tensor digamma(const Tensor&, const std::optional<MemoryConfig>&);
Tensor multigammaln(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor var_hw(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor std_hw(const Tensor&, const std::optional<MemoryConfig>&);
Tensor normalize_hw(const Tensor&, const std::optional<MemoryConfig>&);
Tensor glu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor reglu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor geglu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor swiglu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor tril(const Tensor&, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor triu(const Tensor&, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor polygamma(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor logical_not_(const Tensor&, const std::optional<MemoryConfig>&);
Tensor normalize_global(const Tensor&, const std::optional<MemoryConfig>&);
Tensor frac(const Tensor&, const std::optional<MemoryConfig>&);

}  // namespace ttnn::operations::unary

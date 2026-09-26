// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {

// Rank-2 measurements also apply when extra leading dimensions are all unit
// dimensions because they describe the same matrix workload.
bool has_only_unit_front_dimensions(const ttnn::Shape& shape) noexcept;

// Observe and resolve one public matmul-family call. On a selected hit, the
// complete registry-owned parameter object replaces `parameters` atomically.
// Every observation, inspection, or materialization failure leaves it
// unchanged.
void try_apply_registry_parameters(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    bool has_bias,
    CallSemantics call_semantics,
    ttnn::prim::MatmulParams& parameters,
    const std::optional<ttnn::Tensor>& optional_output_tensor);

}  // namespace ttnn::operations::matmul::registry

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

// The opt-in model selector's entry point (ttnn.CONFIG.matmul_model_config, README.md in this directory).
namespace ttnn::operations::matmul::model {

// The program config the matmul config model picks for this matmul, or nullopt to keep the default selection.
// Called once per matmul by ttnn.matmul / ttnn.linear / ttnn.addmm (bound_matmul) with the inputs and normalized
// attributes (create_matmul_attributes) they pass to the device op, and only when no program config was given.
// Returns nullopt, without loading anything, when ttnn.CONFIG.matmul_model_config is off. Never throws.
std::optional<MatmulProgramConfig> model_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const std::optional<Tensor>& optional_output_tensor,
    const ttnn::prim::MatmulParams& attributes);

// What the model selector did on this thread's last matmul: "model: <program config>" or "default: <reason>".
std::string last_pick();

}  // namespace ttnn::operations::matmul::model

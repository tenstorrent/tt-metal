// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "deepseek_moe_gate_device_operation_types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

// Tensor-backed circular buffers. create_descriptor and override_runtime_arguments both use these indices.
inline constexpr uint8_t kInputCb = 0;
inline constexpr uint8_t kBiasCb = 1;
inline constexpr uint8_t kOutputCb = 2;
inline constexpr uint8_t kInputIndicesCb = 3;
inline constexpr uint8_t kOutputIndicesCb = 4;

// Build executable program descriptor from current tensors / scalars (cache miss only).
tt::tt_metal::ProgramDescriptor build_moe_gate_program_descriptor(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs);

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate

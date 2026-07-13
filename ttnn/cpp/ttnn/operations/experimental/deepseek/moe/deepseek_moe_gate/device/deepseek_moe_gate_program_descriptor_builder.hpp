// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "deepseek_moe_gate_device_operation_types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

// Build executable program descriptor from current tensors / scalars (used on cache hit and miss).
tt::tt_metal::ProgramDescriptor build_moe_gate_program_descriptor(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs);

// Structural hash excluding buffer addresses (matches generic_op program-cache semantics).
[[nodiscard]] std::uint64_t hash_moe_gate_program_structure(const tt::tt_metal::ProgramDescriptor&);

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate

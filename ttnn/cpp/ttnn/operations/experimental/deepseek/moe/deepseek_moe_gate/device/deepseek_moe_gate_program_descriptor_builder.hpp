// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "deepseek_moe_gate_device_operation_types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

// Build executable program descriptor from current tensors / scalars (cache miss only).
tt::tt_metal::ProgramDescriptor build_moe_gate_program_descriptor(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs);

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "generalized_moe_gate_device_operation_types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate {

// Build the executable program descriptor from the current tensors / scalars. Called by create_program
// (cache miss); the program-cache key is a cheap structural hash in compute_program_hash, so this is no
// longer on the cache-hit path.
tt::tt_metal::ProgramDescriptor build_moe_gate_program_descriptor(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs);

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate

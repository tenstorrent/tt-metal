// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Correctness/device-resource gate for prim::reshape_codegen, consulted by the
// free function's routing, by validate_on_program_cache_miss, and by
// reshape_force_codegen. `input` is the tensor as it will actually be handed to
// the prim (post any upstream layout/rank normalization the caller has already
// done); `output_logical_shape`/`output_padded_shape` and `output_mem_config`
// describe the requested result.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by ttnn::reshape only, never by validate and never by
// reshape_force_codegen.
bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::reshape_codegen

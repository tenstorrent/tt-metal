// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::pad_codegen {

enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the `implementation` kwarg. TT_FATALs on an unrecognized value.
ImplementationSelector parse_implementation(const std::string& implementation);

// Correctness / device-resource-feasibility gate, transcribed from
// codegen_pad.py's invalidate_vector plus the manifest's hand-authored
// sharded-in/out and per-layout L1-floor cases. `output_padded_shape` and
// `front` are in the same 4D-padded logical space ttnn::pad computes before
// dispatch (N, C, H, W order); `output_mem_config` is the resolved output
// memory config. Never consults performance -- see is_demoted for that.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& output_padded_shape,
    const std::array<uint32_t, 4>& front,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by the auto branch only, never by validate or forced codegen.
// v1 stub: always false. Demotion analysis is out of scope for this port.
bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& output_padded_shape,
    const std::array<uint32_t, 4>& front,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::pad_codegen

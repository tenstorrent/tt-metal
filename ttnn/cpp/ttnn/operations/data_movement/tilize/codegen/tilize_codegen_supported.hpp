// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace ttnn::operations::data_movement::tilize_codegen {

enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the `implementation` kwarg. TT_FATALs (message per the manifest's
// invalid_selector_message) on an unrecognized value.
ImplementationSelector parse_implementation(const std::string& implementation);

// Mirrors codegen builder_utils/arch.ArchConfig.allocatable_l1_bytes: the CB budget every
// codegen builder plans against (physical per-core L1 minus the allocator's reserved base),
// minus the additional 128 KiB code/stack reserve spec.py's _L1_RESERVE subtracts on top.
// Queried from the device so the gate and the factory can never disagree with what the
// allocator will actually hand out.
uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device);

// Correctness-only: true iff one of the codegen tilize builders (row / block / 2D-column) can
// produce a bit-exact result for this (input, output_mem_config, output_dtype) case. Consulted
// by the free function's forced "codegen" branch and by prim::tilize_codegen's validate --
// never gated on performance.
bool supported_by_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

// Every codegen builder places work over the full compute-with-storage grid and has no
// single-core / sub-core-grid variant, and none of them implement the low-perf single-buffered
// legacy mode's semantics beyond what the row builder's use_low_perf branch already covers. False
// means the case must go to native (under "auto") or be rejected outright (under a forced
// "codegen").
bool supported_execution_controls(
    bool use_multicore, bool use_low_perf, const std::optional<CoreRangeSet>& sub_core_grids);

// Perf-only: true for enumerated in-scope cases where codegen is correct but does not beat
// native on device. Consulted ONLY by the free function's "auto" branch, alongside
// supported_by_codegen(); never by validate, and never under forced implementation="codegen".
bool is_demoted(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

}  // namespace ttnn::operations::data_movement::tilize_codegen

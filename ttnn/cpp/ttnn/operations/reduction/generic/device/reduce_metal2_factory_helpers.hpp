// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared identifiers and helpers for the Metal 2.0 reduction program factories
// (H, W, single-core HW, Welford W/H/HW). DFB id strings are reused across
// factories because the ported kernels reference the same dfb::input /
// dfb::scaler / dfb::output names; the strings only need to be unique within
// a single ProgramSpec.
//
// Kept in a header (inline linkage) rather than per-cpp anonymous namespaces
// because the reduction CMake target uses Unity builds and anonymous
// namespaces from different .cpp files merge in the unity TU.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include <tt-metalium/data_types.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>

namespace ttnn::prim {

namespace metal2_reduce_helpers {
namespace m2 = tt::tt_metal::experimental::metal2_host_api;
}  // namespace metal2_reduce_helpers

// DFB id strings shared by every reduction factory. Identical strings are
// intentional because every factory reuses the same kernel-side names
// (dfb::in_dfb, dfb::scaler_dfb, etc.); the strings only need to be unique
// within a single ProgramSpec.
inline constexpr const char* IN_DFB = "in_dfb";
inline constexpr const char* SCALER_DFB = "scaler_dfb";
inline constexpr const char* OUT_DFB = "out_dfb";
inline constexpr const char* ACC_DFB = "acc_dfb";    // Reduce-negate only
inline constexpr const char* INEG_DFB = "ineg_dfb";  // Reduce-negate only
// Welford-specific DFB ids
inline constexpr const char* WELFORD_VAR_DFB = "var_dfb";            // W only
inline constexpr const char* WELFORD_SCALED_DFB = "scaled_dfb";      // W only
inline constexpr const char* WELFORD_PARTIAL_DFB = "partial_dfb";    // HW only
inline constexpr const char* WELFORD_COMBINED_DFB = "combined_dfb";  // HW only
// Width-sharded H has a second input DFB built on the input shard buffer
inline constexpr const char* IN_SHARD_DFB = "in_shard_dfb";

// TensorParameter ids
inline constexpr const char* INPUT_TENSOR = "input";
inline constexpr const char* OUTPUT_TENSOR = "output";

// Convert a legacy std::map<string, string> defines table into the metal2
// KernelSpec::CompilerOptions::Defines vector form.
inline metal2_reduce_helpers::m2::KernelSpec::CompilerOptions::Defines DefinesFromMap(
    const std::map<std::string, std::string>& src) {
    metal2_reduce_helpers::m2::KernelSpec::CompilerOptions::Defines out;
    out.reserve(src.size());
    for (const auto& [k, v] : src) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace ttnn::prim

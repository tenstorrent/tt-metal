// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared identifiers and helpers for the Metal 2.0 reduction program factories
// (W, HW, Welford). The DFB id strings below are reused across factories because
// the kernels reference the same dfb::input / dfb::scaler / dfb::output names;
// DefinesFromMap converts the legacy std::map<string, string> reduce-defines
// table into the metal2 KernelSpec::CompilerOptions::Defines vector format.
//
// Kept in a header (inline linkage) rather than per-cpp anonymous namespaces because
// the reduction CMake target uses Unity builds and anonymous namespaces from
// different .cpp files merge in the unity TU and collide.

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

// DFB id strings shared by every reduction factory. Identical strings are intentional
// because every factory reuses the same kernel-side names (dfb::input, dfb::scaler,
// etc.) — the strings only need to be unique within a single ProgramSpec.
inline constexpr const char* INPUT_DFB = "input";
inline constexpr const char* SCALER_DFB = "scaler";
inline constexpr const char* OUTPUT_DFB = "output";
inline constexpr const char* ACC_DFB = "acc";    // negate-only
inline constexpr const char* INEG_DFB = "ineg";  // negate-only

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

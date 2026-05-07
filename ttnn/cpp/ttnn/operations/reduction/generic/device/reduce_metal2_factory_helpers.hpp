// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the Metal 2.0 reduction program factories (W, HW, ...).
// Each factory builds its own ProgramSpec; the helpers below collapse the boilerplate
// for declaring DataflowBufferSpec entries, binding DFBs to kernels, and converting
// the legacy std::map<string, string> defines into the metal2 KernelSpec format.
//
// These are kept in a header (with `inline` linkage) rather than per-cpp anonymous
// namespaces because reduction's CMake target uses Unity builds — anonymous namespaces
// from different .cpp files merge in the unity TU and collide.

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

inline metal2_reduce_helpers::m2::DataflowBufferSpec MakeDFB(
    const std::string& name,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::DataFormat data_format,
    const tt::tt_metal::Tile& tile) {
    metal2_reduce_helpers::m2::DataflowBufferSpec dfb;
    dfb.unique_id = name;
    dfb.entry_size = entry_size;
    dfb.num_entries = num_entries;
    dfb.data_format_metadata = data_format;
    dfb.tile_format_metadata = tile;
    return dfb;
}

// Variant of MakeDFB for "intra-tensix" DFBs — DFBs whose producer and consumer
// kernels are the same kernel (e.g., the negate-path acc/ineg scratch buffers
// produced and consumed by the compute kernel, or welford W's var/scaled
// scratch buffers). The Metal 2.0 framework asserts `!enable_implicit_sync` for
// intra-tensix DFBs in `dataflow_buffer.cpp` (ISR-based credit flow only makes
// sense across distinct producer/consumer kernels), so we must disable it.
inline metal2_reduce_helpers::m2::DataflowBufferSpec MakeIntraDFB(
    const std::string& name,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::DataFormat data_format,
    const tt::tt_metal::Tile& tile) {
    auto dfb = MakeDFB(name, entry_size, num_entries, data_format, tile);
    dfb.disable_implicit_sync = true;
    return dfb;
}

inline void BindDFB(
    metal2_reduce_helpers::m2::KernelSpec& kernel,
    const std::string& dfb_name,
    const std::string& accessor_name,
    metal2_reduce_helpers::m2::KernelSpec::DFBEndpointType endpoint_type) {
    kernel.dfb_bindings.push_back(metal2_reduce_helpers::m2::KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = endpoint_type,
        .access_pattern = metal2_reduce_helpers::m2::DFBAccessPattern::STRIDED,
    });
}

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

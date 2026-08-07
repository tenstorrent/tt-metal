// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include <tt_stl/assert.hpp>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>

// Helpers shared by the two moreh_layer_norm_backward program factories (input_grad and
// gamma_beta_grad). They live in a header because the factories are unity-built, where per-file
// anonymous namespaces merge and duplicate definitions collide.
namespace ttnn::operations::moreh::moreh_layer_norm_backward_metal2 {

namespace m2 = tt::tt_metal::experimental;

// Build a DataflowBufferSpec holding num_tiles entries of one tile each. Every buffer in this op is
// a whole number of standard 32x32 tiles, so entry_size is always one tile.
inline m2::DataflowBufferSpec make_dfb(m2::DFBSpecName unique_id, uint32_t num_tiles, tt::DataFormat data_format) {
    return m2::DataflowBufferSpec{
        .unique_id = std::move(unique_id),
        .entry_size = tt::tile_size(data_format),
        .num_entries = num_tiles,
        .data_format_metadata = data_format,
    };
}

// Bind a buffer the kernel both fills and drains, which is how every compute-private intermediate in
// this op is used. One accessor name serves both directions, so the kernel builds a single object.
inline void bind_self_loop(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name) {
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = accessor_name,
        .endpoint_type = m2::DFBEndpointType::PRODUCER,
    });
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::move(accessor_name),
        .endpoint_type = m2::DFBEndpointType::CONSUMER,
    });
}

// Bind one end of an ordinary producer -> consumer buffer.
inline void bind_dfb(
    m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name, m2::DFBEndpointType endpoint) {
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::move(accessor_name),
        .endpoint_type = endpoint,
    });
}

// Record that a buffer is unpacked through the SrcA/SrcB register files.
//
// A compute kernel with the 32-bit Dest register enabled has to state a mode for every Float32 buffer
// it consumes: at that combination the choice between SrcA/B and Dest is a real precision and
// throughput tradeoff, so there is no implicit default to fall back on. Call this once per such
// buffer, under the same condition that binds the buffer — never from a loop over the kernel's
// bindings, which would hide a buffer whose mode nobody actually chose.
inline void unpack_via_src(m2::ComputeGen1Config& compute_config, const m2::DFBSpecName& dfb) {
    compute_config.unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
}

// Resolve the Gen1 alternative of a compute hardware config so per-DFB fields can be set on it.
//
// A compute hardware config holds one generation's settings, and this op only ever builds the Gen1
// (Wormhole / Blackhole) alternative: `to_compute_hardware_config` returns the Gen2 alternative on
// Quasar, and nothing here populates the Gen2-only fields or makes the Quasar-specific choices those
// need. Say so plainly rather than letting the std::get raise std::bad_variant_access.
inline m2::ComputeGen1Config& gen1_compute_config(m2::ComputeHardwareConfig& config) {
    TT_FATAL(
        std::holds_alternative<m2::ComputeGen1Config>(config),
        "moreh_layer_norm_backward builds Gen1 (Wormhole / Blackhole) compute configs only; this device "
        "reports a different generation, which this op does not support yet.");
    return std::get<m2::ComputeGen1Config>(config);
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_metal2

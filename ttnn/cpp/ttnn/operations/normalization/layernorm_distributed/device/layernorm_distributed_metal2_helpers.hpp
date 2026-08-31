// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>

// Helpers shared by the layernorm_distributed program factories. They live in a header because the
// factories are unity-built into one translation unit, where per-file anonymous namespaces merge and
// duplicate definitions collide.
namespace ttnn::prim::layernorm_distributed_metal2 {

namespace m2 = tt::tt_metal::experimental;

// Build a DataflowBufferSpec holding num_tiles entries of one tile each.
inline m2::DataflowBufferSpec make_dfb(
    m2::DFBSpecName unique_id, uint32_t num_tiles, uint32_t tile_size, tt::DataFormat data_format) {
    return m2::DataflowBufferSpec{
        .unique_id = std::move(unique_id),
        .entry_size = tile_size,
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

// Record that a buffer is unpacked through the SrcA/SrcB register files.
//
// A compute kernel with the 32-bit Dest register enabled has to state a mode for every Float32 buffer
// it consumes: at that combination the choice between SrcA/B and Dest is a real precision and
// throughput tradeoff, so there is no implicit default to fall back on. A program factory should call
// this once per such buffer, under the same condition that binds the buffer.
inline void unpack_via_src(m2::ComputeHardwareConfig& compute_config, const m2::DFBSpecName& dfb) {
    compute_config.unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
}

// Record that a buffer is unpacked straight into the Dest register, keeping full 32-bit precision.
inline void unpack_via_dest(m2::ComputeHardwareConfig& compute_config, const m2::DFBSpecName& dfb) {
    compute_config.unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToDest);
}

}  // namespace ttnn::prim::layernorm_distributed_metal2

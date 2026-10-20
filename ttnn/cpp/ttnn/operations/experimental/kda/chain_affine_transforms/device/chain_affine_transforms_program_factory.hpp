// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/tt_backend_api_types.hpp>

#include "chain_affine_transforms_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Per-core DFBs: initial, product, state and out each hold one FP32 [K, V] state block, and the BF16 A and B rows
// are double-buffered so the next step's rows stream while compute applies this one.
inline constexpr uint32_t chain_affine_transforms_state_blocks = 4;
inline constexpr uint32_t chain_affine_transforms_transform_buffers = 2;

inline uint64_t chain_affine_transforms_l1_bytes(uint64_t key_tiles, uint64_t value_tiles) {
    return (chain_affine_transforms_state_blocks * key_tiles * value_tiles * tt::tile_size(tt::DataFormat::Float32)) +
           (chain_affine_transforms_transform_buffers * key_tiles * (key_tiles + value_tiles) *
            tt::tile_size(tt::DataFormat::Float16_b));
}

struct ChainAffineTransformsProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const ChainAffineTransformsParams&,
        const ChainAffineTransformsInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim

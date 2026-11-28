// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "block_gather_device_operation_types.hpp"

namespace ttnn::operations::block_gather {

// Python GATHER_PARTS: a row's blocks split into this many work units when K allows.
constexpr uint32_t kGatherParts = 4;

// Work units per row: the first of (4, 2, 1) with p == 1 or (K % p == 0 and (K / p) % 16 == 0), so a part's ids
// start at a 64-byte aligned offset (Blackhole's DRAM read alignment). Shared by the factory and the validator.
uint32_t gather_parts(uint32_t k);

// CB bytes: 64 + 2 span, span = round_up(4 KP, 64) + round_up(2 B KP, 64) + (2 B == 64 ? 0 : 64 KP), KP = K / parts.
uint32_t gather_scratch_bytes(uint32_t k, uint32_t block);

// One program on the (grid_x x grid_y) rectangle from (0, 0): scratch CB, the kernel on both RISCs
// (kernel_index::HALF0 reader config, HALF1 writer config), common args scores / ids / out addresses and rows.
struct BlockGatherProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BlockGatherParams& attrs, const BlockGatherInputs& t, Tensor& output);

    // Writes the three addresses of both kernels' common args; rows and everything else the key pins.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const BlockGatherParams& attrs,
        const BlockGatherInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::block_gather

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tt_stl/assert.hpp>

#include <umd/device/types/arch.hpp>

#include "ttnn/operations/wavelet/planner/execution_plan.hpp"

namespace ttnn::operations::wavelet {

struct ArchitecturePolicy {
    tt::ARCH architecture{tt::ARCH::Invalid};
    WorkspaceLayout ilwt_layout{WorkspaceLayout::kRowMajor};
    bool inverse_scale_inline{true};
    bool final_interleave_direct{false};
    bool compact_2d_reader{false};
    uint64_t inverse_2d_coordination_penalty_cycles_per_core{0};
    uint32_t l1_scratch_bytes{0};
};

[[nodiscard]] inline ArchitecturePolicy make_architecture_policy(
    const tt::ARCH architecture, const std::optional<WorkspaceLayout> ilwt_layout_override = std::nullopt) {
    constexpr uint64_t kBlackholeInverse2DCoordinationPenaltyCyclesPerCore = 6'000;
    switch (architecture) {
        case tt::ARCH::WORMHOLE_B0:
            return ArchitecturePolicy{
                .architecture = architecture,
                .ilwt_layout = ilwt_layout_override.value_or(WorkspaceLayout::kRowMajor),
                .inverse_scale_inline = true,
                .final_interleave_direct = false,
                .compact_2d_reader = true,
                .inverse_2d_coordination_penalty_cycles_per_core = 0,
                .l1_scratch_bytes = 0,
            };
        case tt::ARCH::BLACKHOLE: {
            const WorkspaceLayout layout = ilwt_layout_override.value_or(WorkspaceLayout::kTileNative);
            return ArchitecturePolicy{
                .architecture = architecture,
                .ilwt_layout = layout,
                .inverse_scale_inline = true,
                .final_interleave_direct = layout == WorkspaceLayout::kTileNative,
                // Complex boundary specializations can exceed the shared 69 KiB
                // Tensix kernel-config window even for modest route counts. Keep
                // helper bodies out of line on both supported architectures.
                .compact_2d_reader = true,
                .inverse_2d_coordination_penalty_cycles_per_core = kBlackholeInverse2DCoordinationPenaltyCyclesPerCore,
                .l1_scratch_bytes = 0,
            };
        }
        default: TT_THROW("tt-wavelet supports only Wormhole B0 and Blackhole, got {}", architecture);
    }
}

}  // namespace ttnn::operations::wavelet

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::wavelet::planner_cost_model {

// Dimensionless relative weights used only to rank planner candidates
inline constexpr uint64_t kCoreStartup = 30'000;
inline constexpr uint64_t kInitialElement = 12;
inline constexpr uint64_t kRouteConfigAndSync = 3'700;
inline constexpr uint64_t kExactStaging = 900;
inline constexpr uint64_t kOneAxisShiftedStaging = 7'000;
inline constexpr uint64_t kGenericStaging = 9'000;
inline constexpr uint64_t kUnknownStaging = 70'000;
inline constexpr uint64_t kFullTilePersistence = 1'200;
inline constexpr uint64_t kNonStencilRoute = 8'000;
inline constexpr uint64_t kVerticalStencilBase = 16'000;
inline constexpr uint64_t kVerticalStencilTap = 2'500;
inline constexpr uint64_t kHorizontalStencilBase = 12'000;
inline constexpr uint64_t kHorizontalStencilTap = 1'800;
inline constexpr uint64_t kFragmentedTerminalTile = 80'000;
inline constexpr uint64_t kInterleavedTerminalTile = 80'000;
inline constexpr uint64_t kTiledTerminalTile = 1'200;
inline constexpr long double kFewerCoresCostRatio = 0.90L;
inline constexpr long double kMoreCoresCostRatio = 1.10L;

}  // namespace ttnn::operations::wavelet::planner_cost_model

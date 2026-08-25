// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <span>

#include "agmm_registry_descriptor.hpp"

namespace ttnn::experimental::all_gather_minimal_matmul_registry::generated {

inline constexpr compact::TableMetadata kMetadata{};
inline constexpr std::array<compact::EntryDescriptor, 0> kEntries{};

inline constexpr const compact::TableMetadata& metadata() noexcept { return kMetadata; }
inline constexpr std::span<const compact::EntryDescriptor> entries() noexcept { return kEntries; }

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry::generated

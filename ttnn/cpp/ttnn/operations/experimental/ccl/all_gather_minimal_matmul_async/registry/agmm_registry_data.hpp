// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <span>

#include "agmm_registry_descriptor.hpp"

namespace ttnn::experimental::all_gather_minimal_matmul_registry::generated {

inline constexpr compact::TableLock kLock{};
inline constexpr std::array<compact::EntryDescriptor, 0> kEntries{};

static_assert(compact::validate_table_lock(kLock, kEntries) == compact::TableValidationStatus::Empty);

inline constexpr const compact::TableLock& lock() noexcept { return kLock; }
inline constexpr std::span<const compact::EntryDescriptor> entries() noexcept { return kEntries; }

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry::generated

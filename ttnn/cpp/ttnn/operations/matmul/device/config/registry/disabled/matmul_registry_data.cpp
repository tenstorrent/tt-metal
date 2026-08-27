// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_registry_data.hpp"

#include <array>

namespace ttnn::operations::matmul::registry::generated {
namespace {

// A build that explicitly disables Python-backed artifact generation must not
// claim the checked lock or any compatibility evidence.  Empty spans make
// every runtime mode fall through before compatibility or device attestation.
constexpr compact::TableMetadata kMetadata{};
constexpr std::array<compact::EntryDescriptor, 0> kEntries{};
constexpr std::array<compact::ProgramConfigExactEntry, 0> kProgramConfigExactEntries{};
constexpr std::array<compact::ProgramConfigGbdtModel, 0> kOnlineModels{};

}  // namespace

const compact::TableMetadata& metadata() noexcept { return kMetadata; }
std::span<const compact::EntryDescriptor> entries() noexcept { return kEntries; }
compact::ExactIndex index() noexcept { return compact::ExactIndex{kEntries}; }
std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries() noexcept {
    return kProgramConfigExactEntries;
}
std::span<const compact::ProgramConfigGbdtModel> online_models() noexcept { return kOnlineModels; }

}  // namespace ttnn::operations::matmul::registry::generated

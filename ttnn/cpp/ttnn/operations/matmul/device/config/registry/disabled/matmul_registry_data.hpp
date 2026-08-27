// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

#include "ttnn/operations/matmul/device/config/registry/matmul_program_config_model.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace ttnn::operations::matmul::registry::generated {

const compact::TableMetadata& metadata() noexcept;
std::span<const compact::EntryDescriptor> entries() noexcept;
compact::ExactIndex index() noexcept;
std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries() noexcept;
std::span<const compact::ProgramConfigGbdtModel> online_models() noexcept;

}  // namespace ttnn::operations::matmul::registry::generated

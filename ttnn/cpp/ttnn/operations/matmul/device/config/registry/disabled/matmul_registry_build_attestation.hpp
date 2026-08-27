// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace ttnn::operations::matmul::registry::generated_build {

inline constexpr std::uint16_t kAttestationSchemaVersion = 1;
inline constexpr compact::Sha256 kActualSemanticSourceSha256{};
inline constexpr compact::Sha256 kActualBuildIdentitySha256{};

}  // namespace ttnn::operations::matmul::registry::generated_build

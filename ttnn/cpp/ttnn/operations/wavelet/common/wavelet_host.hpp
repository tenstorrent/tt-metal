// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/generated/wavelet_schemes/scheme_catalog.hpp"

namespace ttnn::operations::wavelet {

[[nodiscard]] BoundaryMode boundary_mode_from_string(std::string_view name);

[[nodiscard]] SchemeId scheme_id_from_string(std::string_view name);

[[nodiscard]] const SchemeInfo& scheme_info(SchemeId id);

[[nodiscard]] uint32_t dwt_coefficient_length(uint32_t input_length, SchemeId id);

}  // namespace ttnn::operations::wavelet

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/common/wavelet_host.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::wavelet {
namespace {

[[nodiscard]] constexpr bool parse_boundary_mode(const std::string_view name, BoundaryMode& mode) noexcept {
    if (name == "zero") {
        mode = BoundaryMode::kZero;
    } else if (name == "constant") {
        mode = BoundaryMode::kConstant;
    } else if (name == "symmetric") {
        mode = BoundaryMode::kSymmetric;
    } else if (name == "periodic") {
        mode = BoundaryMode::kPeriodic;
    } else if (name == "antisymmetric") {
        mode = BoundaryMode::kAntisymmetric;
    } else if (name == "smooth") {
        mode = BoundaryMode::kSmooth;
    } else if (name == "antireflect") {
        mode = BoundaryMode::kAntireflect;
    } else if (name == "reflect") {
        mode = BoundaryMode::kReflect;
    } else {
        return false;
    }
    return true;
}

}  // namespace

BoundaryMode boundary_mode_from_string(const std::string_view name) {
    BoundaryMode mode{};
    TT_FATAL(
        parse_boundary_mode(name, mode),
        "Unsupported wavelet boundary mode '{}'; expected: zero, constant, symmetric, reflect, periodic, smooth, "
        "antisymmetric, or antireflect",
        name);
    return mode;
}

SchemeId scheme_id_from_string(const std::string_view name) {
    const SchemeId id = scheme_id(name);
    TT_FATAL(id != SchemeId::kUnknown, "Unsupported wavelet scheme '{}'", name);
    return id;
}

const SchemeInfo& scheme_info(const SchemeId id) {
    const size_t index = static_cast<size_t>(id);
    TT_FATAL(index < kSchemeInfos.size(), "Invalid wavelet scheme id {}", index);
    return kSchemeInfos[index];
}

uint32_t dwt_coefficient_length(const uint32_t input_length, const SchemeId id) {
    TT_FATAL(input_length > 0, "DWT input length must be greater than zero");
    const auto& info = scheme_info(id);
    const uint64_t coefficient_length =
        (static_cast<uint64_t>(input_length) + static_cast<uint64_t>(info.tap_size) - 1U) / 2U;
    TT_FATAL(
        coefficient_length <= std::numeric_limits<uint32_t>::max(),
        "DWT coefficient length {} exceeds the uint32 range",
        coefficient_length);
    return static_cast<uint32_t>(coefficient_length);
}

}  // namespace ttnn::operations::wavelet

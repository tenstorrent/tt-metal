// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

#include "ttnn/operations/wavelet/wavelet_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

[[nodiscard]] uint32_t dwt_coeff_len(uint32_t input_length, std::string_view wavelet);

std::tuple<Tensor, Tensor> dwt(
    const Tensor& input,
    std::string_view wavelet,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::tuple<Tensor, Tensor>>& output_tensors = std::nullopt);

Tensor idwt(
    const Tensor& approximation,
    const Tensor& detail,
    std::string_view wavelet,
    uint32_t original_length,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

std::tuple<Tensor, Tensor, Tensor, Tensor> dwt_2d(
    const Tensor& input,
    std::string_view wavelet,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::array<Tensor, 4>>& output_tensors = std::nullopt);

Tensor idwt_2d(
    const Tensor& ll,
    const Tensor& lh,
    const Tensor& hl,
    const Tensor& hh,
    std::string_view wavelet,
    const WaveletOutputShape2D& output_shape,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn

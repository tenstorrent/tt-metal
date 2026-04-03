
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

Tensor real(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Tensor imag(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Tensor angle(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Tensor is_imag(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Tensor is_real(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
ComplexTensor conj(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
ComplexTensor polar(
    const ComplexTensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn

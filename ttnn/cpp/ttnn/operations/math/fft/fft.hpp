// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

ComplexTensor fft(
    const ComplexTensor& input_tensor,
    int64_t dim = -1,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

ComplexTensor fft(
    const Tensor& input_tensor,
    int64_t dim = -1,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

ComplexTensor ifft(
    const ComplexTensor& input_tensor,
    int64_t dim = -1,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

ComplexTensor ifft(
    const Tensor& input_tensor,
    int64_t dim = -1,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn

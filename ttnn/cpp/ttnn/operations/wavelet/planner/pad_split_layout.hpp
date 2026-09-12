// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/common/signal.hpp"

namespace ttnn::operations::wavelet {

struct Pad1DConfig {
    BoundaryMode mode{BoundaryMode::kSymmetric};
    uint32_t left{0};
    uint32_t right{0};
};

struct PadSplit1DLayout {
    SignalBuffer input{};
    Pad1DConfig pad_config{};
    Signal output{};

    [[nodiscard]] constexpr size_t padded_length() const noexcept {
        return input.length + static_cast<size_t>(pad_config.left) + static_cast<size_t>(pad_config.right);
    }
};

[[nodiscard]] constexpr PadSplit1DLayout make_pad_split_1d_layout(
    const SignalBuffer& input, const Pad1DConfig config) noexcept {
    const size_t length = input.length + static_cast<size_t>(config.left) + static_cast<size_t>(config.right);

    return PadSplit1DLayout{.input = input, .pad_config = config, .output = make_split_signal(input, length)};
}

}  // namespace ttnn::operations::wavelet

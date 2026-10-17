// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "tt-metalium/math.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"

namespace ttnn::operations::wavelet {

struct SignalBuffer {
    size_t length{0};
    uint32_t stick_width{kStickWidth};
    uint32_t element_size_bytes{sizeof(float)};
};

struct Signal {
    SignalBuffer even;
    SignalBuffer odd;
};

[[nodiscard]] constexpr Signal make_split_signal(const SignalBuffer& input, const size_t source_length) noexcept {
    const size_t even_len = tt::div_up(source_length, size_t{2});
    const size_t odd_len = source_length / 2;

    return Signal{
        .even =
            SignalBuffer{
                .length = even_len, .stick_width = input.stick_width, .element_size_bytes = input.element_size_bytes},
        .odd = SignalBuffer{
            .length = odd_len, .stick_width = input.stick_width, .element_size_bytes = input.element_size_bytes}};
}

}  // namespace ttnn::operations::wavelet

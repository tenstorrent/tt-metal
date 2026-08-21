// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "ttnn/operations/wavelet/common/storage_contract.hpp"

namespace ttnn::operations::wavelet {

[[nodiscard]] constexpr size_t ceil_div(const size_t numerator, const size_t denominator) noexcept {
    return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

[[nodiscard]] constexpr size_t round_up(const std::size_t value, const std::size_t alignment) noexcept {
    return alignment == 0 ? value : ceil_div(value, alignment) * alignment;
}

// Describes a logical 1D signal stored as complete row-major sticks.
// The FP32 path uses 32 elements (128 bytes) per stick. `length` excludes
// unused lanes in the final stick; those lanes are unspecified.
struct SignalBuffer {
    uint64_t dram_address{0};
    size_t length{0};
    uint32_t stick_width{kStickWidth};
    uint32_t element_size_bytes{sizeof(float)};

    [[nodiscard]] constexpr size_t stick_count() const noexcept {
        return ceil_div(length, static_cast<size_t>(stick_width));
    }

    [[nodiscard]] constexpr uint32_t stick_bytes() const noexcept { return stick_width * element_size_bytes; }

    [[nodiscard]] constexpr uint32_t aligned_stick_bytes(
        const uint32_t alignment = kStorageAlignmentBytes) const noexcept {
        return static_cast<uint32_t>(round_up(static_cast<size_t>(stick_bytes()), alignment));
    }

    [[nodiscard]] constexpr size_t physical_nbytes(const uint32_t alignment = kStorageAlignmentBytes) const noexcept {
        return stick_count() * static_cast<size_t>(aligned_stick_bytes(alignment));
    }
};

// Even and odd streams produced by splitting a 1D signal.
struct Signal {
    SignalBuffer even;
    SignalBuffer odd;
};

[[nodiscard]] constexpr Signal make_split_signal(
    const SignalBuffer& input, const size_t source_length, const uint64_t even_addr, const uint64_t odd_addr) noexcept {
    const size_t even_len = ceil_div(source_length, size_t{2});
    const size_t odd_len = source_length / 2;

    return Signal{
        .even =
            SignalBuffer{
                .dram_address = even_addr,
                .length = even_len,
                .stick_width = input.stick_width,
                .element_size_bytes = input.element_size_bytes},
        .odd = SignalBuffer{
            .dram_address = odd_addr,
            .length = odd_len,
            .stick_width = input.stick_width,
            .element_size_bytes = input.element_size_bytes}};
}

}  // namespace ttnn::operations::wavelet

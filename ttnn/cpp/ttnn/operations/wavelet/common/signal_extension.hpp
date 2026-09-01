// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/operations/wavelet/common/boundary.hpp"

#define WAVELET_EXTENSION_ALWI inline __attribute__((always_inline))

namespace ttnn::operations::wavelet {

enum class ExtensionOperation : uint8_t {
    kZero = 0,
    kSample = 1,
    kNegatedSample = 2,
    kSmooth = 3,
    kAntireflect = 4,
};

struct ExtendedIndexI32 {
    uint32_t source_index{0};
    uint32_t auxiliary_index{0};
    uint32_t distance{0};
    int32_t period_quotient{0};
    ExtensionOperation operation{ExtensionOperation::kZero};
    bool reflected{false};
};

// Compact antireflect descriptor for device paths whose coordinates and
// logical lengths are already constrained to signed 32-bit values.
struct AntireflectIndexI32 {
    uint32_t source_index{0};
    int32_t period_quotient{0};
    bool reflected{false};
    bool affine{false};
};

struct SmoothIndexI32 {
    uint32_t source_index{0};
    uint32_t auxiliary_index{0};
    uint32_t distance{0};
    bool affine{false};
};

[[nodiscard]] constexpr WAVELET_EXTENSION_ALWI uint32_t
extension_positive_mod_i32(const int32_t index, const uint32_t period) noexcept {
    if (index >= 0) {
        return static_cast<uint32_t>(index) % period;
    }
    const uint32_t magnitude = 0U - static_cast<uint32_t>(index);
    const uint32_t tail = magnitude % period;
    return tail == 0 ? 0U : period - tail;
}

[[nodiscard]] constexpr WAVELET_EXTENSION_ALWI uint32_t
make_symmetric_index_i32(const int32_t index, const uint32_t length) noexcept {
    if (length == 0) {
        return 0;
    }
    if (index >= 0 && static_cast<uint32_t>(index) < length) {
        return static_cast<uint32_t>(index);
    }
    const uint32_t period = 2U * length;
    const uint32_t phase = extension_positive_mod_i32(index, period);
    return phase < length ? phase : period - 1U - phase;
}

[[nodiscard]] inline __attribute__((noinline)) AntireflectIndexI32
make_antireflect_index_i32(const int32_t index, const uint32_t length) noexcept {
    if (length == 0) {
        return {};
    }
    if (index >= 0 && static_cast<uint32_t>(index) < length) {
        return AntireflectIndexI32{
            .source_index = static_cast<uint32_t>(index),
        };
    }
    if (length == 1) {
        return {};
    }

    // Public 2D entry points limit logical dimensions to INT32_MAX. Therefore
    // 2 * (length - 1) fits uint32_t even when it does not fit int32_t.
    const uint32_t last = length - 1U;
    const uint32_t period = 2U * last;
    int32_t quotient = 0;
    uint32_t remainder = 0;
    if (index >= 0) {
        const uint32_t positive_index = static_cast<uint32_t>(index);
        quotient = static_cast<int32_t>(positive_index / period);
        remainder = positive_index % period;
    } else {
        // Unsigned negation is well-defined for INT32_MIN.
        const uint32_t magnitude = 0U - static_cast<uint32_t>(index);
        const uint32_t whole_periods = magnitude / period;
        const uint32_t tail = magnitude % period;
        quotient = -static_cast<int32_t>(whole_periods);
        if (tail != 0) {
            --quotient;
            remainder = period - tail;
        }
    }
    const bool reflected = remainder > last;
    return AntireflectIndexI32{
        .source_index = reflected ? period - remainder : remainder,
        .period_quotient = quotient,
        .reflected = reflected,
        .affine = true,
    };
}

[[nodiscard]] inline __attribute__((noinline)) SmoothIndexI32
make_smooth_index_i32(const int32_t index, const uint32_t length) noexcept {
    if (length == 0) {
        return {};
    }
    if (index >= 0 && static_cast<uint32_t>(index) < length) {
        return SmoothIndexI32{
            .source_index = static_cast<uint32_t>(index),
        };
    }
    if (length == 1) {
        return {};
    }
    const bool left = index < 0;
    return SmoothIndexI32{
        .source_index = left ? 0U : length - 1U,
        .auxiliary_index = left ? 1U : length - 2U,
        .distance = left ? 0U - static_cast<uint32_t>(index) : static_cast<uint32_t>(index) - (length - 1U),
        .affine = true,
    };
}

namespace detail {

template <BoundaryMode Mode>
struct BoundaryExtensionPolicy;

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kZero> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32 make_i32(const int32_t, const uint32_t) noexcept {
        return {};
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kConstant> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        return ExtendedIndexI32{
            .source_index = index < 0 ? 0U : length - 1U,
            .operation = ExtensionOperation::kSample,
        };
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kPeriodic> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        return ExtendedIndexI32{
            .source_index = extension_positive_mod_i32(index, length),
            .operation = ExtensionOperation::kSample,
        };
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kSymmetric> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        return ExtendedIndexI32{
            .source_index = make_symmetric_index_i32(index, length),
            .operation = ExtensionOperation::kSample,
        };
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kAntisymmetric> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        // The sign/reversal pattern repeats every two segments. Since length is
        // at most INT32_MAX, the unsigned period cannot overflow.
        const uint32_t period = 2U * length;
        const uint32_t phase = extension_positive_mod_i32(index, period);
        const bool negated = phase >= length;
        return ExtendedIndexI32{
            .source_index = negated ? period - 1U - phase : phase,
            .operation = negated ? ExtensionOperation::kNegatedSample : ExtensionOperation::kSample,
        };
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kSmooth> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        const SmoothIndexI32 smooth = make_smooth_index_i32(index, length);
        return ExtendedIndexI32{
            .source_index = smooth.source_index,
            .auxiliary_index = smooth.auxiliary_index,
            .distance = smooth.distance,
            .operation = smooth.affine ? ExtensionOperation::kSmooth : ExtensionOperation::kSample,
        };
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kReflect> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        if (length == 1) {
            return ExtendedIndexI32{
                .source_index = 0,
                .operation = ExtensionOperation::kSample,
            };
        }
        const uint32_t last = length - 1U;
        const uint32_t period = 2U * last;
        const uint32_t phase = extension_positive_mod_i32(index, period);
        return ExtendedIndexI32{
            .source_index = phase <= last ? phase : period - phase,
            .operation = ExtensionOperation::kSample,
        };
    }
};

template <>
struct BoundaryExtensionPolicy<BoundaryMode::kAntireflect> {
    [[nodiscard]] static WAVELET_EXTENSION_ALWI ExtendedIndexI32
    make_i32(const int32_t index, const uint32_t length) noexcept {
        const AntireflectIndexI32 antireflect = make_antireflect_index_i32(index, length);
        return ExtendedIndexI32{
            .source_index = antireflect.source_index,
            .period_quotient = antireflect.period_quotient,
            .operation = antireflect.affine ? ExtensionOperation::kAntireflect : ExtensionOperation::kSample,
            .reflected = antireflect.reflected,
        };
    }
};

}  // namespace detail

template <BoundaryMode Mode>
[[nodiscard]] WAVELET_EXTENSION_ALWI ExtendedIndexI32
make_extended_index_i32(const int32_t index, const uint32_t length) noexcept {
    static_assert(is_supported_lwt_boundary_mode(Mode), "Unsupported padding mode");

    if (length == 0) {
        return {};
    }
    if (index >= 0 && static_cast<uint32_t>(index) < length) {
        return ExtendedIndexI32{
            .source_index = static_cast<uint32_t>(index),
            .operation = ExtensionOperation::kSample,
        };
    }
    return detail::BoundaryExtensionPolicy<Mode>::make_i32(index, length);
}

template <typename SourceReader>
[[nodiscard]] WAVELET_EXTENSION_ALWI float evaluate_antireflect_index_i32(
    const AntireflectIndexI32& extended, const uint32_t length, const SourceReader& read_source) noexcept {
    if (!extended.affine) {
        return read_source(extended.source_index);
    }
    const float source = read_source(extended.source_index);
    const float first = read_source(0);
    const float last = read_source(length - 1U);
    const float base = extended.reflected ? 2.0F * last - source : source;
    return base + (static_cast<float>(extended.period_quotient) * 2.0F) * (last - first);
}

template <typename SourceReader>
[[nodiscard]] WAVELET_EXTENSION_ALWI float evaluate_smooth_index_i32(
    const SmoothIndexI32& extended, const SourceReader& read_source) noexcept {
    if (!extended.affine) {
        return read_source(extended.source_index);
    }
    const float edge = read_source(extended.source_index);
    const float neighbor = read_source(extended.auxiliary_index);
    return edge + static_cast<float>(extended.distance) * (edge - neighbor);
}

template <BoundaryMode Mode, typename SourceReader>
[[nodiscard]] WAVELET_EXTENSION_ALWI float evaluate_extended_index_i32(
    const ExtendedIndexI32& extended, const uint32_t length, const SourceReader& read_source) noexcept {
    static_assert(is_supported_lwt_boundary_mode(Mode), "Unsupported signal-extension mode");
    if constexpr (Mode == BoundaryMode::kZero) {
        return extended.operation == ExtensionOperation::kZero ? 0.0F : read_source(extended.source_index);
    } else if constexpr (
        Mode == BoundaryMode::kConstant || Mode == BoundaryMode::kSymmetric || Mode == BoundaryMode::kReflect ||
        Mode == BoundaryMode::kPeriodic) {
        return read_source(extended.source_index);
    } else if constexpr (Mode == BoundaryMode::kAntisymmetric) {
        const float source = read_source(extended.source_index);
        return extended.operation == ExtensionOperation::kNegatedSample ? -source : source;
    } else if constexpr (Mode == BoundaryMode::kSmooth) {
        if (extended.operation == ExtensionOperation::kSample) {
            return read_source(extended.source_index);
        }
        const float edge = read_source(extended.source_index);
        const float neighbor = read_source(extended.auxiliary_index);
        return edge + static_cast<float>(extended.distance) * (edge - neighbor);
    } else {
        static_assert(Mode == BoundaryMode::kAntireflect);
        if (extended.operation == ExtensionOperation::kSample) {
            return read_source(extended.source_index);
        }
        const float source = read_source(extended.source_index);
        const float first = read_source(0);
        const float last = read_source(length - 1U);
        const float base = extended.reflected ? 2.0F * last - source : source;
        return base + (static_cast<float>(extended.period_quotient) * 2.0F) * (last - first);
    }
}

template <typename SourceIndexConsumer>
WAVELET_EXTENSION_ALWI void visit_antireflect_source_indices_i32(
    const AntireflectIndexI32& extended, const uint32_t length, const SourceIndexConsumer& consume) noexcept {
    consume(extended.source_index);
    if (extended.affine) {
        consume(0);
        consume(length - 1U);
    }
}

template <typename SourceIndexConsumer>
WAVELET_EXTENSION_ALWI void visit_smooth_source_indices_i32(
    const SmoothIndexI32& extended, const SourceIndexConsumer& consume) noexcept {
    consume(extended.source_index);
    if (extended.affine) {
        consume(extended.auxiliary_index);
    }
}

template <BoundaryMode Mode, typename SourceIndexConsumer>
WAVELET_EXTENSION_ALWI void visit_extended_source_indices_i32(
    const ExtendedIndexI32& extended, const uint32_t length, const SourceIndexConsumer& consume) noexcept {
    static_assert(is_supported_lwt_boundary_mode(Mode), "Unsupported signal-extension mode");
    if constexpr (Mode == BoundaryMode::kZero) {
        if (extended.operation != ExtensionOperation::kZero) {
            consume(extended.source_index);
        }
    } else if constexpr (Mode == BoundaryMode::kSmooth) {
        consume(extended.source_index);
        if (extended.operation != ExtensionOperation::kSample) {
            consume(extended.auxiliary_index);
        }
    } else if constexpr (Mode == BoundaryMode::kAntireflect) {
        consume(extended.source_index);
        if (extended.operation != ExtensionOperation::kSample) {
            consume(0);
            consume(length - 1U);
        }
    } else {
        consume(extended.source_index);
    }
}

}  // namespace ttnn::operations::wavelet

#undef WAVELET_EXTENSION_ALWI

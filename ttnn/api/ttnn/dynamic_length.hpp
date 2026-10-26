// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::dynamic_length {

// The valid (unpadded) extent of one tensor axis: a host scalar, or a single-valued UINT32/INT32 device tensor
// that the kernels read at run time, so the value can change between replays of one captured trace.
using ValidLength = std::variant<std::size_t, Tensor>;

struct SplitLength {
    // The host value; equals the capacity when the length lives in `tensor`, so host-side sizing sees the worst case.
    std::size_t value = 0;
    std::optional<Tensor> tensor;
};

SplitLength split(const ValidLength& length, std::size_t capacity);

// An axis whose valid length may differ from the extent the program was built for. The program is sized for
// `capacity`, so a host length above it is refused rather than run.
struct DynamicAxis {
    static constexpr std::size_t kUnbounded = std::numeric_limits<std::size_t>::max();

    std::string_view name;
    std::size_t capacity = kUnbounded;
    // Host value. Ignored when `tensor` is set: the live value is only known on device, so neither capacity nor
    // alignment is checked for it. Keeping it within both is the caller's contract (or the kernel's, if it clamps).
    std::size_t value = 0;
    const Tensor* tensor = nullptr;
    std::size_t alignment = 1;
};

// Checks every host length against its capacity and alignment, and checks that a length tensor is a single
// allocated UINT32/INT32 value on device (its value is not read). Throws naming `operation_name` and the first
// offending axis.
void validate_axes(std::span<const DynamicAxis> axes, std::string_view operation_name);

}  // namespace ttnn::dynamic_length

namespace ttnn::device_operation {

// Opt-in: the framework validates the declared axes on every launch, on program cache hits as well as misses,
// because an op whose hash drops a length can be handed a cached program sized for a smaller capacity.
template <typename device_operation_t>
concept HasDynamicAxes = requires(
    const typename device_operation_t::operation_attributes_t& attrs,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    {
        device_operation_t::dynamic_axes(attrs, tensor_args)
    } -> std::convertible_to<std::vector<dynamic_length::DynamicAxis>>;
};

}  // namespace ttnn::device_operation

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

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
    // Host value. Always checked against `capacity` and `alignment`, including when `tensor` is set: `split()`
    // stores the capacity there, and a direct caller must not pass a larger placeholder. The word stored in
    // `tensor` is not read on the host. Keeping that live value within capacity and alignment is the caller's
    // contract (or the kernel's, if it clamps). A trace capture additionally requires a scalar axis's value to
    // equal its capacity; see the mesh adapter.
    std::size_t value = 0;
    const Tensor* tensor = nullptr;
    std::size_t alignment = 1;
};

// Checks every host placeholder against its capacity and alignment, whether or not a length tensor is set, and
// checks that a length tensor is a single allocated UINT32/INT32 value on device. The tensor's stored word is not
// read. Throws naming `operation_name` and the first offending axis.
void validate_axes(std::span<const DynamicAxis> axes, std::string_view operation_name);

}  // namespace ttnn::dynamic_length

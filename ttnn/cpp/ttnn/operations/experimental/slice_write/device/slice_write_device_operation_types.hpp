// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct SliceWriteParams {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;

    static constexpr auto attribute_names = std::forward_as_tuple("slice_start", "slice_end", "step");
    auto attribute_values() const { return std::forward_as_tuple(slice_start, slice_end, step); }
};

struct SliceWriteInputs {
    Tensor input;
    Tensor output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "output");
    auto attribute_values() const { return std::forward_as_tuple(input, output); }
};

using ReaderKernelArgs = std::vector<uint32_t>;
using WriterKernelArgs = std::vector<uint32_t>;
using KernelRuntimeArgs = std::pair<ReaderKernelArgs, WriterKernelArgs>;
using SliceWriteRuntimeArgs = std::vector<KernelRuntimeArgs>;

}  // namespace ttnn::experimental::prim

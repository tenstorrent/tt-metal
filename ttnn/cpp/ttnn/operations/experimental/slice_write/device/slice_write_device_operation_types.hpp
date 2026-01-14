// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::slice_write {

struct operation_attributes_t {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;
};

struct tensor_args_t {
    Tensor input;
    Tensor output;
};

using ReaderKernelArgs = std::vector<uint32_t>;
using WriterKernelArgs = std::vector<uint32_t>;
using KernelRuntimeArgs = std::pair<ReaderKernelArgs, WriterKernelArgs>;
using SliceWriteRuntimeArgs = std::vector<KernelRuntimeArgs>;

}  // namespace ttnn::operations::experimental::slice_write

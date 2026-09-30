// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ScatterCodegenParams;
struct ScatterCodegenInputs;

// TILE, full input/src row resident in L1.
struct ScatterCodegenProgramFactoryInterleaved {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// TILE, chunked streaming fallback for rows too wide for the interleaved plan's L1 budget.
struct ScatterCodegenProgramFactoryStreaming {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// ROW_MAJOR fast path.
struct ScatterCodegenProgramFactoryRowMajor {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// ROW_MAJOR, bfloat16 reduction deferred to FP32 arithmetic.
struct ScatterCodegenProgramFactoryBf16ReduceRowMajor {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim

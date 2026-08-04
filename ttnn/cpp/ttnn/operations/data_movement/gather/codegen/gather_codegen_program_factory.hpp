// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GatherCodegenParams;
struct GatherCodegenInputs;

// Row-buffered: full Wt_input row resident in L1 (kernels/gather_reader.cpp, gather_writer.cpp).
struct GatherCodegenProgramFactoryInterleaved {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Per-output-tile split: high parallelism for small Ht (kernels/gather_reader_tiled.cpp,
// gather_writer_tiled.cpp).
struct GatherCodegenProgramFactoryTiled {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Double-buffered streaming: large Wt_input that doesn't fit the row-buffered L1 budget
// (kernels/gather_reader_streaming.cpp, gather_writer_streaming.cpp).
struct GatherCodegenProgramFactoryStreaming {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim

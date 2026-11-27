// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SliceWriteParams {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;
    // A tiled sharded input's program serves every start and output size, and the H and W ends may lie past the
    // output, where the slice is clipped (see SliceWriteTiledShardedInputRuntimeShapesProgramFactory).
    const bool runtime_shapes = false;
};

struct SliceWriteInputs {
    Tensor input;
    Tensor output;
};

using ReaderKernelArgs = std::vector<uint32_t>;
using WriterKernelArgs = std::vector<uint32_t>;
using KernelRuntimeArgs = std::pair<ReaderKernelArgs, WriterKernelArgs>;
using SliceWriteRuntimeArgs = std::vector<KernelRuntimeArgs>;

}  // namespace ttnn::experimental::prim

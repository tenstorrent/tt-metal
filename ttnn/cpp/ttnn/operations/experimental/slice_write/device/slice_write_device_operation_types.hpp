// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SliceWriteParams {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;
};

struct SliceWriteInputs {
    Tensor input;
    Tensor output;
};

using ReaderKernelArgs = std::vector<uint32_t>;
using WriterKernelArgs = std::vector<uint32_t>;
using KernelRuntimeArgs = std::pair<ReaderKernelArgs, WriterKernelArgs>;
using SliceWriteRuntimeArgs = std::vector<KernelRuntimeArgs>;

// RTA 0 of the slice_write writers (and of the interleaved reader) is the base address of the buffer they access.
inline constexpr uint32_t kSliceWriteAddressArgIdx = 0;

// Emplaces one core's runtime args with RTA 0 bound to `buffer` as a Buffer* (so the framework tracks the
// address); every other arg is copied as a plain value. Shared by all three slice_write factories.
inline void emplace_slice_write_runtime_args(
    tt::tt_metal::KernelDescriptor& kernel,
    const tt::tt_metal::CoreCoord& core,
    const std::vector<uint32_t>& args,
    tt::tt_metal::Buffer* buffer) {
    tt::tt_metal::KernelDescriptor::RTArgList runtime_args;
    runtime_args.reserve(args.size());
    for (uint32_t arg_idx = 0; arg_idx < args.size(); ++arg_idx) {
        if (arg_idx == kSliceWriteAddressArgIdx) {
            runtime_args.push_back(buffer);
        } else {
            runtime_args.push_back(args[arg_idx]);
        }
    }
    kernel.emplace_runtime_args(core, runtime_args);
}

}  // namespace ttnn::experimental::prim

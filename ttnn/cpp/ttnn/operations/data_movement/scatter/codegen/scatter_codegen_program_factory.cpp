// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_program_factory.hpp"

#include "scatter_codegen_device_operation.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryInterleaved::create_descriptor(
    const ScatterCodegenParams& /*attributes*/,
    const ScatterCodegenInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    return {};
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryStreaming::create_descriptor(
    const ScatterCodegenParams& /*attributes*/,
    const ScatterCodegenInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    return {};
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryRowMajor::create_descriptor(
    const ScatterCodegenParams& /*attributes*/,
    const ScatterCodegenInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    return {};
}

tt::tt_metal::ProgramDescriptor ScatterCodegenProgramFactoryBf16ReduceRowMajor::create_descriptor(
    const ScatterCodegenParams& /*attributes*/,
    const ScatterCodegenInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    return {};
}

}  // namespace ttnn::prim

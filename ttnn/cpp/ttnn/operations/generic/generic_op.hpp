// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include "ttnn/compiled_program.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// GenericOp exposes everything needed to construct and write an operation on device for the user.
// This includes: cb attributes, data movement attributes, compute attributes, rt args, compile time args.
// Unlike other operations, must create and pass in output tensor with the input tensors.
// See tests/ttnn/unit_tests/gtests/test_generic_op.cpp for some examples.
// The main use case right now is an interface for PyKernel to pass dynamic kernel paths.

// Primary entry point for mesh programs
Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor);

// Convenience entry point for single ProgramDescriptor (SPMD mode)
Tensor generic_op(const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);

namespace experimental {

// Compiles and finalizes the program that generic_op would enqueue, without enqueueing it. Throws if the program does
// not compile or does not fit the kernel-configuration buffer. The prepared workload is inserted into the program
// cache, so the next generic_op with the same tensors and descriptor is a cache hit.
void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor);
void prepare_generic_op(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);

// Compiles and lays out the program that generic_op would enqueue, into a CompiledProgram, without enqueueing it or
// touching the program cache (ttnn::device_operation::compile). In slow dispatch its kernel binaries may outgrow the
// KERNEL_CONFIG window, to run from a kernel config buffer (ttnn::device_operation::create_kernel_config_buffer).
ttnn::device_operation::CompiledProgram compile_generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor);
ttnn::device_operation::CompiledProgram compile_generic_op(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);

// generic_op, launching `compiled` (from compile_generic_op) instead of a program from the program cache: the
// descriptor's runtime arguments, and the buffers of its CBs, are applied to it as a cache hit applies them
// (ttnn::device_operation::run). The descriptor must have the program cache key of the one `compiled` was compiled
// from. `kernel_config_buffer`, when given, holds the Tensix kernel binaries instead of the KERNEL_CONFIG window.
Tensor run_compiled_program(
    ttnn::device_operation::CompiledProgram& compiled,
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor,
    const tt::tt_metal::experimental::KernelConfigBuffer* kernel_config_buffer = nullptr);
Tensor run_compiled_program(
    ttnn::device_operation::CompiledProgram& compiled,
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::ProgramDescriptor& program_descriptor,
    const tt::tt_metal::experimental::KernelConfigBuffer* kernel_config_buffer = nullptr);

}  // namespace experimental

}  // namespace ttnn

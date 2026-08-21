// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include "ttnn/types.hpp"

namespace ttnn {

// GenericOp exposes everything needed to construct and write an operation on device for the user.
// This includes: cb attributes, data movement attributes, compute attributes, rt args, compile time args.
// Unlike other operations, must create and pass in output tensor with the input tensors. Only the
// output is required: a generator program (no tensor read) or a single-tensor in-place program
// passes a one-element io_tensors.
// See tests/ttnn/unit_tests/gtests/test_generic_op.cpp for some examples.
// The main use case right now is an interface for PyKernel to pass dynamic kernel paths.

// Primary entry point for mesh programs
Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor);

// Convenience entry point for single ProgramDescriptor (SPMD mode)
Tensor generic_op(const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);

// Metal 2.0 entry point: a ProgramSpec + ProgramRunArgs instead of a ProgramDescriptor.
//
// tensor_args maps each TensorParameter name to an index into io_tensors. The op builds the real
// TensorArgument table from it, so the pointer-identity requirement on TensorArguments is
// satisfied by construction.
Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::ProgramSpec& spec,
    const tt::tt_metal::experimental::ProgramRunArgs& run_args,
    const tt::tt_metal::experimental::Table<tt::tt_metal::experimental::TensorParamName, uint32_t>& tensor_args);

}  // namespace ttnn

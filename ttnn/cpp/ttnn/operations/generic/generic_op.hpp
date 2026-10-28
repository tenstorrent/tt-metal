// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/distributed/types.hpp"
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

// generic_op on a Metal 2.0 ProgramSpec, which describes everything fixed about a program, and ProgramRunArgs, which
// hold the values of one call. Arguments are named and tensors are bound by name. Experimental while its API may still
// change. See tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py for examples.

using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;

// Fixed: a ProgramSpec per mesh coordinate range.
struct MeshProgramSpec {
    std::vector<std::pair<ttnn::MeshCoordinateRange, ProgramSpec>> programs;
    std::optional<uint64_t> custom_program_hash;
};

// Per call: the run args for each range of the MeshProgramSpec.
struct MeshProgramRunArgs {
    std::vector<std::pair<ttnn::MeshCoordinateRange, ProgramRunArgs>> programs;
};

// Runs the same program on every device that holds the tensors. The output is the last tensor in io_tensors, which
// the caller preallocates, and each tensor parameter binds its own io_tensors entry. The program cache keys on a hash
// of the spec, or on custom_program_hash combined with the names the run args refer to, each kernel's cores and
// its vararg counts.
Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const ProgramSpec& spec,
    const ProgramRunArgs& run_args,
    std::optional<uint64_t> custom_program_hash = std::nullopt);

// Runs each range's program on the devices in that range.
Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const MeshProgramSpec& mesh_program_spec,
    const MeshProgramRunArgs& mesh_run_args);

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const ProgramSpec& spec,
    const ProgramRunArgs& run_args,
    std::optional<uint64_t> custom_program_hash = std::nullopt);

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const MeshProgramSpec& mesh_program_spec,
    const MeshProgramRunArgs& mesh_run_args);

// The hash generic_op keys its program cache on when no custom_program_hash is given (combined with each program's
// mesh range). Argument values, ProgramSpec::name and WorkUnitSpec::name are not part of it.
uint64_t compute_program_spec_hash(const ProgramSpec& spec);

}  // namespace experimental

}  // namespace ttnn

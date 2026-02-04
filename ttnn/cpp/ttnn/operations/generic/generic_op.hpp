// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include "ttnn/decorators.hpp"

namespace ttnn::operations::generic {

// GenericOp exposes everything needed to construct and write an operation on device for the user.
// This includes: cb attributes, data movement attributes, compute attributes, rt args, compile time args.
// Unlike other operations, must create and pass in output tensor with the input tensors.
// See tests/ttnn/unit_tests/gtests/test_generic_op.cpp for some examples.
// The main use case right now is an interface for PyKernel to pass dynamic kernel paths.

struct GenericOp {
    // TODO: #20830 - Split io_tensors into input_tensors and output_tensor properly.

    // Primary entry point for mesh programs
    static Tensor invoke(
        const std::vector<Tensor>& io_tensors,
        const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor);

    // Convenience entry point for single ProgramDescriptor (SPMD mode)
    static Tensor invoke(
        const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);
};  // struct GenericOp

}  // namespace ttnn::operations::generic

namespace ttnn {
constexpr auto generic_op = ttnn::register_operation<"ttnn::generic_op", ttnn::operations::generic::GenericOp>();
}  // namespace ttnn

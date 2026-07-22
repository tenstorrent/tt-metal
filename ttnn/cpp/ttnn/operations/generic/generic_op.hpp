// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
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

// A fixed-address GenericOp prepared as an owned MeshWorkload.
//
// Unlike generic_op(), dispatch() does not enter the TTNN program cache, rebuild
// ProgramDescriptors, or re-apply runtime arguments.  The descriptors are
// consumed once by the constructor and every subsequent dispatch enqueues the
// same MeshWorkload on the command queue selected at construction.
//
// This direct enqueue intentionally bypasses ordinary TTNN graph, inspector,
// and Tracy operation instrumentation. Low-level mesh tracing remains
// available after eager warmup.
//
// io_tensors are retained for the lifetime of the handle so their allocations
// stay live.  Any non-tensor resource represented only by an address in a
// descriptor (for example a GlobalSemaphore) must likewise be retained by the
// caller; the Python binding accepts a resource_owners object for that purpose.
class PreparedGenericOp {
public:
    PreparedGenericOp(
        const std::vector<Tensor>& io_tensors,
        const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor,
        std::optional<std::uint8_t> cq_id = std::nullopt);
    ~PreparedGenericOp();

    PreparedGenericOp(PreparedGenericOp&&) noexcept;
    PreparedGenericOp& operator=(PreparedGenericOp&&) noexcept;
    PreparedGenericOp(const PreparedGenericOp&) = delete;
    PreparedGenericOp& operator=(const PreparedGenericOp&) = delete;

    // Non-blocking enqueue on the command queue pinned at construction. The
    // handle is caller-synchronized and not safe for concurrent host calls.
    void dispatch();

    // Wait for all work on the pinned command queue.  Destruction also drains
    // an outstanding dispatch before releasing fixed-address resources.
    void synchronize();

    const Tensor& output_tensor() const;
    std::uint8_t cq_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ttnn

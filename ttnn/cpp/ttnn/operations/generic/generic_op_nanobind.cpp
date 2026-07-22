// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <string>
#include <memory>
#include <optional>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "generic_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

namespace {

// Binding-only owner wrapper.  PreparedGenericOp owns the fixed tensor handles
// and MeshWorkload; resource_owners pins arbitrary Python-side owners whose
// device addresses may also be embedded in the descriptor (GlobalSemaphores,
// MeshBuffers, etc.).  Declaration order is intentional: prepared_ is destroyed
// first and drains outstanding work before resource_owners_ is released.
class PyPreparedGenericOp {
    nb::object resource_owners_;
    ttnn::PreparedGenericOp prepared_;

public:
    PyPreparedGenericOp(
        const std::vector<Tensor>& io_tensors,
        const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor,
        nb::object resource_owners,
        std::optional<std::uint8_t> cq_id) :
        resource_owners_(std::move(resource_owners)), prepared_(io_tensors, mesh_program_descriptor, cq_id) {}

    void dispatch() { prepared_.dispatch(); }
    void synchronize() { prepared_.synchronize(); }
    Tensor output_tensor() const { return prepared_.output_tensor(); }
    std::uint8_t cq_id() const { return prepared_.cq_id(); }
};

}  // namespace

// Defined in generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);

void bind_generic_operation(nb::module_& mod) {
    std::string doc =
        R"doc(
        Executes a custom operation with user-defined kernels on the device.

        The generic_op provides a flexible interface for constructing and executing custom operations
        on device hardware. It allows specifying custom compute kernels, data movement, and control flow.

        Args:
            io_tensors (List[ttnn.Tensor]): List of input tensors and output tensor(s). Output tensor(s) must be
                pre-allocated and included as the last element(s) in the list.
            program_descriptor (ttnn.ProgramDescriptor or ttnn.MeshProgramDescriptor): Descriptor containing
                kernel specifications, computational buffer configurations, semaphores, and other execution
                parameters. Use ProgramDescriptor for SPMD mode (same program on all devices) or
                MeshProgramDescriptor for explicit per-device control.

        Returns:
            ttnn.Tensor: Handle to the output tensor.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op.py for usage examples
        )doc";

    // Overload for MeshProgramDescriptor (explicit mesh control)
    auto mesh_program_overload = ttnn::overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const tt::tt_metal::experimental::MeshProgramDescriptor&)>(
            &ttnn::generic_op),
        nb::arg("io_tensors"),
        nb::arg("mesh_program_descriptor"));

    // Overload for ProgramDescriptor (SPMD mode - broadcasts to all devices)
    auto program_overload = ttnn::overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const tt::tt_metal::ProgramDescriptor&)>(&ttnn::generic_op),
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"));

    ttnn::bind_function<"generic_op">(mod, doc.c_str(), mesh_program_overload, program_overload);

    nb::class_<PyPreparedGenericOp>(mod, "PreparedGenericOp", R"pbdoc(
        Opaque fixed-address generic-op dispatch handle.

        The MeshProgramDescriptor is converted into an owned MeshWorkload once.
        ``dispatch()`` then re-enqueues that workload without descriptor copies,
        program-cache lookup, or runtime-argument refresh. The command queue is
        pinned when the handle is prepared. Tensor allocations are retained by
        the handle; pass non-tensor address owners through ``resource_owners``.

        Dispatch is asynchronous. ``synchronize()`` drains the pinned queue, and
        destruction drains any outstanding dispatch before releasing resources.
        The handle is caller-synchronized and not safe for concurrent host calls.
        Every tensor coordinate must be covered by exactly one descriptor range.

        This direct enqueue intentionally bypasses normal TTNN graph, inspector,
        and Tracy operation instrumentation. Low-level mesh tracing remains
        supported after eager warmup.
    )pbdoc")
        .def("dispatch", &PyPreparedGenericOp::dispatch, nb::call_guard<nb::gil_scoped_release>())
        .def("synchronize", &PyPreparedGenericOp::synchronize, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("output_tensor", &PyPreparedGenericOp::output_tensor)
        .def_prop_ro("cq_id", &PyPreparedGenericOp::cq_id);

    mod.def(
        "prepare_generic_op",
        [](const std::vector<Tensor>& io_tensors,
           const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor,
           nb::object resource_owners,
           std::optional<std::uint8_t> cq_id) {
            return std::make_unique<PyPreparedGenericOp>(
                io_tensors, mesh_program_descriptor, std::move(resource_owners), cq_id);
        },
        nb::arg("io_tensors"),
        nb::arg("mesh_program_descriptor"),
        nb::kw_only(),
        nb::arg("resource_owners") = nb::none(),
        nb::arg("cq_id") = nb::none(),
        R"pbdoc(
            Prepare an immutable fixed-address ``MeshProgramDescriptor`` for reuse.

            Args:
                io_tensors: Fixed device tensors referenced by the descriptor.
                mesh_program_descriptor: Descriptor consumed once during preparation.
                resource_owners: Optional arbitrary owner object retained until all
                    outstanding dispatches drain. Use this for GlobalSemaphores or
                    other resources represented only by raw device addresses.
                cq_id: Optional command queue id. Defaults to the current thread's
                    command queue and remains pinned for the handle lifetime.

            This direct enqueue intentionally bypasses normal TTNN graph,
            inspector, and Tracy operation instrumentation. Low-level mesh
            tracing remains supported after eager warmup.
        )pbdoc");

    mod.def(
        "compute_program_descriptor_hash",
        &compute_program_descriptor_hash,
        nb::arg("program_descriptor"),
        R"pbdoc(
            Compute structural hash of a ProgramDescriptor.

            Hashes kernel sources, compile-time args, core ranges, CB structure,
            and semaphores. Excludes runtime arg values and buffer addresses,
            making it suitable as a cache key for structural equivalence.
        )pbdoc");
}

}  // namespace ttnn::operations::generic

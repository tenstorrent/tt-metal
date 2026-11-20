// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_nanobind.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/ranges.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "device/generic_program_spec_device_operation.hpp"
#include "generic_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/program_spec.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using ttnn::experimental::MeshProgramRunArgs;
using ttnn::experimental::MeshProgramSpec;
using ttnn::experimental::ProgramRunArgs;
using ttnn::experimental::ProgramSpec;

bool on_device(const Tensor& tensor) { return is_device_tensor(tensor) && tensor.is_allocated(); }

// On a one-device mesh every view holds the only device, so the views need no reading there.
bool on_one_device_mesh(const std::vector<Tensor>& io_tensors) {
    const auto entry = std::ranges::find_if(io_tensors, on_device);
    return entry == io_tensors.end() || entry->device()->num_devices() == 1;
}

// The devices that a ProgramSpec-form call runs its program on, as generic_op works them out.
std::vector<MeshCoordinate> program_devices(const std::vector<Tensor>& io_tensors) {
    // A local, not a temporary in the loop header: ranges() refers into the set.
    const auto ranges = ttnn::prim::program_spec_ranges(io_tensors);
    std::vector<MeshCoordinate> devices;
    for (const auto& range : ranges.ranges()) {
        std::ranges::copy(range, std::back_inserter(devices));
    }
    return devices;
}

// A tensor parameter name for a message, with each NUL character written as \0: a NUL would end the message on its
// way to Python, and put a raw NUL byte in logs.
std::string printable(std::string_view name) {
    std::string text;
    text.reserve(name.size());
    for (const char character : name) {
        if (character == '\0') {
            text += "\\0";
        } else {
            text += character;
        }
    }
    return text;
}

// The devices that the program of one range of a MeshProgramSpec runs on: those of the range on the mesh. It walks the
// mesh rather than the range, so a range far past the mesh costs no more than one inside it; generic_op reports a range
// that reaches past the mesh or has another dimension.
std::vector<MeshCoordinate> devices_in(const ttnn::MeshCoordinateRange& range, const ttnn::MeshShape& mesh_shape) {
    std::vector<MeshCoordinate> devices;
    for (const auto& device : ttnn::MeshCoordinateRange(mesh_shape)) {
        if (device.dims() == range.dims() && range.contains(device)) {
            devices.push_back(device);
        }
    }
    return devices;
}

// Per-device views of one tensor, such as those ttnn.get_device_tensors returns, share its MeshTensor, and Metal's
// tensor arguments keep only the MeshTensor: on each device a program runs on, it reads that device's part of the
// tensor. So each tensor that run args made in Python bind to a declared tensor parameter must hold every device that
// the run args are for, or the program would read, on a device the tensor lacks, data the tensor does not hold. A
// tensor that io_tensors lists holds them all, since generic_op rejects any other call; the check catches a per-device
// view that misses one, whatever else is wrong with the call. A tensor under a name that no program declares is never
// read, and one on every device of the mesh holds them all, so neither needs the check; one that io_tensors lists, or
// that is not allocated on a device, is left to generic_op, which checks it on every call and names it. Metal plans to
// take a MeshTensorView as a tensor argument too (tensor_parameter.hpp), which would let it see which devices a view
// holds.
bool holds_every_device(const Tensor& tensor) {
    return !on_device(tensor) || tensor.device_storage().is_uniform_storage();
}

bool listed(const std::vector<Tensor>& io_tensors, const Tensor& tensor) {
    const auto coords = tensor.device_storage().get_coords();
    return std::ranges::any_of(io_tensors, [&](const Tensor& entry) {
        return on_device(entry) && &entry.mesh_tensor() == &tensor.mesh_tensor() &&
               std::ranges::equal(entry.device_storage().get_coords(), coords);
    });
}

bool declares(const ProgramSpec& spec, std::string_view name) {
    return std::ranges::any_of(
        spec.tensor_parameters, [&](const auto& parameter) { return *parameter.unique_id == name; });
}

void check_tensor_holds(std::string_view name, const Tensor& tensor, const std::vector<MeshCoordinate>& devices) {
    const auto held = tensor.device_storage().get_coords();
    const auto missing = std::ranges::find_if(
        devices, [&](const MeshCoordinate& device) { return std::ranges::find(held, device) == held.end(); });
    TT_FATAL(
        missing == devices.end(),
        "generic_op: tensor_args['{}'] holds devices {}, but these run args are for device {}, which it does not hold. "
        "On each device it runs on, a program reads that device's part of a tensor, so to read the data of a "
        "per-device view, run the program only on the view's devices, for example by listing only those devices' "
        "views in io_tensors; binding the whole tensor instead makes each device read its own part, not the view's",
        printable(name),
        fmt::format("{}", fmt::join(held, ", ")),
        *missing);
}

// The checks run with the GIL held, before the call releases it (the overloads are gil_held_overload_t), and read
// the tensors that the Python run args keep without copying them. The visitors capture one reference each, which
// std::function holds without allocating.
void check_views(const std::vector<Tensor>& io_tensors, const ProgramSpec& spec, const ProgramRunArgs& run_args) {
    if (on_one_device_mesh(io_tensors)) {
        return;
    }
    struct Check {
        const std::vector<Tensor>& io_tensors;
        const ProgramSpec& spec;
        // Worked out once per call, and only for a tensor that is not on every device.
        std::optional<std::vector<MeshCoordinate>> devices;
    } check{io_tensors, spec, std::nullopt};
    ttnn::program_spec::for_each_python_tensor_arg(run_args, [&check](std::string_view name, const Tensor& tensor) {
        if (holds_every_device(tensor) || !declares(check.spec, name) || listed(check.io_tensors, tensor)) {
            return;
        }
        if (!check.devices) {
            check.devices = program_devices(check.io_tensors);
        }
        check_tensor_holds(name, tensor, *check.devices);
    });
}

// The run args of each range are for the program on that range (generic_op pairs them with the spec's program for the
// same range, in any order), so each is checked against its own range and the names that program declares. Run args
// for a range with no program never run, and generic_op rejects them.
void check_views(
    const std::vector<Tensor>& io_tensors, const MeshProgramSpec& spec, const MeshProgramRunArgs& run_args) {
    if (on_one_device_mesh(io_tensors)) {
        return;
    }
    struct Check {
        const std::vector<Tensor>& io_tensors;
        const MeshProgramSpec& spec;
        const MeshProgramRunArgs& run_args;
        const ttnn::MeshShape& mesh_shape;
    } check{io_tensors, spec, run_args, std::ranges::find_if(io_tensors, on_device)->device()->shape()};
    ttnn::program_spec::for_each_python_tensor_arg(
        run_args, [&check](std::size_t program, std::string_view name, const Tensor& tensor) {
            const auto& range = check.run_args.programs[program].first;
            const auto spec = std::ranges::find_if(
                check.spec.programs, [&](const auto& candidate) { return candidate.first == range; });
            if (holds_every_device(tensor) || spec == check.spec.programs.end() || !declares(spec->second, name) ||
                listed(check.io_tensors, tensor)) {
                return;
            }
            check_tensor_holds(name, tensor, devices_in(range, check.mesh_shape));
        });
}

// The Metal 2.0 forms as Python calls them: each checks the tensor views while it holds the GIL, then releases it for
// the call itself, as bind_function does for other overloads.
Tensor generic_op_from_python(
    const std::vector<Tensor>& io_tensors, const ProgramSpec& spec, const ProgramRunArgs& run_args) {
    check_views(io_tensors, spec, run_args);
    const nb::gil_scoped_release release;
    return ttnn::experimental::generic_op(io_tensors, spec, run_args);
}

Tensor generic_op_from_python(
    const std::vector<Tensor>& io_tensors, const MeshProgramSpec& spec, const MeshProgramRunArgs& run_args) {
    check_views(io_tensors, spec, run_args);
    const nb::gil_scoped_release release;
    return ttnn::experimental::generic_op(io_tensors, spec, run_args);
}

void prepare_generic_op_from_python(
    const std::vector<Tensor>& io_tensors, const ProgramSpec& spec, const ProgramRunArgs& run_args) {
    check_views(io_tensors, spec, run_args);
    const nb::gil_scoped_release release;
    ttnn::experimental::prepare_generic_op(io_tensors, spec, run_args);
}

void prepare_generic_op_from_python(
    const std::vector<Tensor>& io_tensors, const MeshProgramSpec& spec, const MeshProgramRunArgs& run_args) {
    check_views(io_tensors, spec, run_args);
    const nb::gil_scoped_release release;
    ttnn::experimental::prepare_generic_op(io_tensors, spec, run_args);
}

}  // namespace

// Defined in generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);

// The paragraphs that the Note sections of both generic_op forms' docstrings share, after each form's own first
// paragraph.
constexpr const char* OUTPUT_LABEL_FIX = R"doc(
            After a call that raises, or a ``ttnn.experimental.prepare_generic_op`` call, the label may have changed but
            the program has not run, so give back the label the output had before the call with
            ``output.update_tensor_topology(topology)``, which moves no data. Save that label first with
            ``output.tensor_topology()``, which returns a copy.

            If the program places the output's data otherwise, such as the same data on every device from a sharded
            input, or different data on each device from replicated inputs, fix the label the same way after each call
            that runs it. When every tensor in io_tensors is on every device of the mesh, build the label that ttnn
            gives its own tensors with that placement, from the output's devices::

                coords = output.tensor_topology().mesh_coords()
                shape = output.device().shape
                # The same data on every device, as ttnn.allocate_tensor_on_device labels a tensor
                # (ttnn.from_torch gives it the 1-D form, MeshShape([len(coords)]) with one Replicate):
                same = ttnn.TensorTopology(shape, [ttnn.PlacementReplicate()] * shape.dims(), coords)
                # Split along tensor dimension dim, device by device, as ttnn.shard_tensor_to_mesh_mapper does:
                split = ttnn.TensorTopology(ttnn.MeshShape([len(coords)]), [ttnn.PlacementShard(dim)], coords)

            A label can also give each mesh axis its own placement: ``shape`` with one placement for each of its
            axes, as ``ttnn.create_mesh_mapper`` labels a tensor for a ``ttnn.MeshMapperConfig`` that does not
            override the mesh shape. Split each tensor dimension across one mesh axis at most: ttnn's ops relabel any
            other axis it is split across as replicated. Tensors on part of the mesh, and data that no label
            describes, such as a partial sum on each device, are not covered here. Fix another tensor's label the
            same way if the program changes how its data is placed.
)doc";

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
            ttnn.Tensor: Handle to the output tensor, the last entry of io_tensors. When the tensors are not all on
            the same devices, it holds only its parts on the devices of the tensor in io_tensors that is on the
            fewest devices.

        Note:
            The output's tensor topology is a label of how its data is placed across the mesh: its mesh placements and
            distribution shape. As for most ops, each generic_op call merges this label from the labels of all the
            tensors in io_tensors, the output's own included, so a tensor sharded along a mesh axis usually makes the
            output sharded along it. Other tensors in io_tensors keep their labels. The label changes before the program
            is compiled and before some of generic_op's checks, so a call that raises can still change it.
)doc";
    doc += OUTPUT_LABEL_FIX;
    doc += R"doc(
        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op.py for usage examples
        )doc";

    // Overload for MeshProgramDescriptor (explicit mesh control)
    auto mesh_program_overload = ttnn::overload_t(
        static_cast<Tensor (*)(
            const std::vector<Tensor>&, const tt::tt_metal::experimental::MeshProgramDescriptor&)>(&ttnn::generic_op),
        nb::arg("io_tensors"),
        nb::arg("mesh_program_descriptor"));

    // Overload for ProgramDescriptor (SPMD mode - broadcasts to all devices)
    auto program_overload = ttnn::overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const tt::tt_metal::ProgramDescriptor&)>(
            &ttnn::generic_op),
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"));

    ttnn::bind_function<"generic_op">(mod, doc.c_str(), mesh_program_overload, program_overload);

    mod.def(
        "compute_program_descriptor_hash",
        &compute_program_descriptor_hash,
        nb::arg("program_descriptor"),
        R"pbdoc(
            Compute structural hash of a ProgramDescriptor.

            Hashes kernel sources, compile-time args, core ranges, CB structure,
            and semaphores. Excludes runtime arg values and buffer addresses,
            making it suitable as a cache key for structural equivalence. The
            one exception is a global CB's addresses, which are hashed because a
            cache hit does not re-point them.
        )pbdoc");
}

void bind_generic_operation_preparation(nb::module_& mod) {
    const char* doc =
        R"doc(
        Compiles and finalizes the program that a generic_op call with the same arguments would enqueue, without
        enqueueing it: ``ttnn.generic_op`` for a ProgramDescriptor, ``ttnn.experimental.generic_op`` for a
        ProgramSpec.

        Use it to find out whether a program compiles and fits the device's kernel-configuration buffer before
        committing resources to it. The prepared workload is inserted into the program cache, so the next
        generic_op call with the same tensors and program is a cache hit. A failed preparation leaves the program
        cache unchanged, with one exception: when a ProgramSpec is already cached, its run args are applied to the
        cached program first, so DFB size overrides stay in effect even if the preparation then fails.

        Like the generic_op call it prepares, it changes the output's tensor topology: the label of how the data of the
        output, the last entry of io_tensors, is placed across the mesh. A failed preparation can leave the label
        changed too. The note of ttnn.generic_op or ttnn.experimental.generic_op says how to give the old label back.

        Args:
            io_tensors (List[ttnn.Tensor]): The tensors the generic_op call would receive.
            program_descriptor (ttnn.ProgramDescriptor): The program to prepare in SPMD mode (same program on all
                devices).
            mesh_program_descriptor (ttnn.MeshProgramDescriptor): The per-device programs to prepare, for explicit
                per-device control. Pass exactly one of ``program_descriptor`` and ``mesh_program_descriptor``, or
                pass ``spec`` and ``run_args`` instead.
            spec (ttnn.experimental.ProgramSpec or ttnn.experimental.MeshProgramSpec): The Metal 2.0 program, as for
                ``ttnn.experimental.generic_op``.
            run_args (ttnn.experimental.ProgramRunArgs or ttnn.experimental.MeshProgramRunArgs): Its run args, with
                real tensors.

        Raises:
            RuntimeError: If a kernel does not compile, if the program configuration does not fit the
                kernel-configuration buffer, or if the tensors' MeshDevice has no local devices. With a ProgramSpec,
                also for each error ``ttnn.experimental.generic_op`` raises.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op_preparation.py and
            test_generic_op_program_spec.py for usage examples
        )doc";

    auto mesh_program_overload = ttnn::overload_t(
        static_cast<void (*)(const std::vector<Tensor>&, const tt::tt_metal::experimental::MeshProgramDescriptor&)>(
            &ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("mesh_program_descriptor"));

    auto program_overload = ttnn::overload_t(
        static_cast<void (*)(const std::vector<Tensor>&, const tt::tt_metal::ProgramDescriptor&)>(
            &ttnn::experimental::prepare_generic_op),
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"));

    // The Metal 2.0 forms join this binding: a second prepare_generic_op in the module would replace the first.
    auto program_spec_overload = ttnn::gil_held_overload_t(
        static_cast<void (*)(
            const std::vector<Tensor>&,
            const ttnn::experimental::ProgramSpec&,
            const ttnn::experimental::ProgramRunArgs&)>(&prepare_generic_op_from_python),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    auto mesh_program_spec_overload = ttnn::gil_held_overload_t(
        static_cast<void (*)(
            const std::vector<Tensor>&,
            const ttnn::experimental::MeshProgramSpec&,
            const ttnn::experimental::MeshProgramRunArgs&)>(&prepare_generic_op_from_python),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    ttnn::bind_function<"prepare_generic_op", "ttnn.experimental.">(
        mod, doc, mesh_program_overload, program_overload, program_spec_overload, mesh_program_spec_overload);
}

void bind_generic_operation_program_spec(nb::module_& mod) {
    const char* doc = R"doc(
        Runs a Metal 2.0 program, given as a ttnn.experimental.ProgramSpec (everything fixed about the program) and
        a ttnn.experimental.ProgramRunArgs (the values of this call), with named arguments and tensors bound by name.
        Experimental: the API may still change.

        The first call with a given program compiles its kernels; later calls with the same program write only the
        new argument values. Every call checks the run args against the spec: each kernel gets exactly its declared
        arguments, each array with its declared length, each tensor parameter gets a tensor from io_tensors that
        matches its spec, and each DFB size override names a DFB of the spec, at most once and with no size of 0.
        Whether every core got its per-core values is checked on later calls only when
        ttnn.CONFIG.validate_program_args is on; with it off, a core that a later call leaves out keeps the values it
        was given before.

        Args:
            io_tensors (List[ttnn.Tensor]): The tensors the program uses, with the preallocated output last. Each
                tensor parameter binds its own entry, so a tensor bound to two parameters is listed twice. The list
                may hold only the output.
            spec (ttnn.experimental.ProgramSpec or ttnn.experimental.MeshProgramSpec): The program, run on the
                devices of the tensor in io_tensors that is on the fewest devices (every other tensor must be on them
                too), or a program for each mesh coordinate range.
            run_args (ttnn.experimental.ProgramRunArgs or ttnn.experimental.MeshProgramRunArgs): The values of this
                call, or of each range's program. A tensor argument that is another view of a tensor io_tensors lists
                (they share one MeshTensor), such as one device's view from ttnn.get_device_tensors, must hold every
                device its program runs on, since on each device a program reads that device's part of the tensor:
                bind a one-device view only in a program that runs on that device alone.

        Returns:
            ttnn.Tensor: The output tensor, the last entry of io_tensors. When the tensors are not all on the same
            devices, it holds only its parts on the devices of the tensor in io_tensors that is on the fewest
            devices.

        Note:
            The output's tensor topology is a label of how its data is placed across the mesh: its mesh placements
            and distribution shape. As for most ops, each generic_op or prepare_generic_op call merges this label
            from the labels of all the tensors in io_tensors, the output's own included, so a tensor sharded along a
            mesh axis usually makes the output sharded along it. Other tensors in io_tensors keep their labels. The
            label changes before the program is compiled and before some of the call's checks, such as those of the
            run args against the spec, so a call that raises can still change it.
)doc";
    const char* doc_after_note = R"doc(
        Raises:
            RuntimeError: If the run args do not match the spec, naming the kernel, argument, tensor parameter or
                DFB at fault; if a tensor is on another MeshDevice or misses a device the program runs on; if the
                MeshProgramSpec has no programs, its ranges overlap or reach outside the mesh, or the run args' ranges
                differ from the spec's; if a tensor argument misses a device its program runs on, such as a
                per-device view; if io_tensors is empty or holds a tensor that is not allocated on a device, or
                one allocated per core; if Metal rejects the run args, such as a DFB size it cannot take; or, on a
                first call, if Metal rejects the spec, a kernel does not compile or the program does not fit the
                kernel-configuration buffer.

        Example:
            Refer to tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py for usage examples
        )doc";
    const std::string full_doc = std::string(doc) + OUTPUT_LABEL_FIX + doc_after_note;

    auto program_spec_overload = ttnn::gil_held_overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const ProgramSpec&, const ProgramRunArgs&)>(
            &generic_op_from_python),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    auto mesh_program_spec_overload = ttnn::gil_held_overload_t(
        static_cast<Tensor (*)(const std::vector<Tensor>&, const MeshProgramSpec&, const MeshProgramRunArgs&)>(
            &generic_op_from_python),
        nb::arg("io_tensors"),
        nb::arg("spec"),
        nb::arg("run_args"));

    ttnn::bind_function<"generic_op", "ttnn.experimental.">(
        mod, full_doc.c_str(), program_spec_overload, mesh_program_spec_overload);

    mod.def(
        "compute_program_spec_hash",
        &ttnn::experimental::compute_program_spec_hash,
        nb::arg("spec"),
        R"doc(
        The hash of a ttnn.experimental.ProgramSpec that ttnn.experimental.generic_op keys its program cache on,
        combined with each program's mesh range. Runtime argument values, ProgramSpec.name and WorkUnitSpec.name are
        not part of it, and tensor parameter specs are hashed under their relaxations.
        )doc");
}

}  // namespace ttnn::operations::generic

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compiled_program.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <tt-metalium/experimental/kernel_config_buffer.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/compiled_program.hpp"

namespace ttnn::compiled_program {

namespace {

tt::tt_metal::experimental::KernelConfigBuffer create_kernel_config_buffer(
    ttnn::MeshDevice& mesh_device, ttnn::device_operation::CompiledProgram& compiled, bool bottom_up) {
    return ttnn::device_operation::create_kernel_config_buffer(mesh_device, compiled, bottom_up);
}

}  // namespace

void py_module(nb::module_& mod) {
    nb::class_<ttnn::device_operation::CompiledProgram>(mod, "CompiledProgram", R"doc(
            An operation's program, compiled and laid out for one set of arguments, outside the program cache.
            Run it with arguments that have the same program cache key, optionally from a kernel config buffer.
            Made by an op's compile function, such as ttnn.experimental.compile_generic_op, and run by its run
            function, such as ttnn.experimental.run_compiled_program. Experimental API; may change.
        )doc")
        .def_prop_ro(
            "operation_name",
            &ttnn::device_operation::CompiledProgram::operation_name,
            "The device operation it was compiled from.");

    nb::class_<tt::tt_metal::experimental::KernelConfigBuffer>(mod, "KernelConfigBuffer", R"doc(
            L1 that a compiled program's Tensix kernel binaries run from, at a per-core address, instead of the
            KERNEL_CONFIG window. The rest of the kernel configuration stays in the window. Slow dispatch only.
            Experimental API; may change.
        )doc")
        .def_prop_ro(
            "max_size",
            [](const tt::tt_metal::experimental::KernelConfigBuffer& self) { return self.max_size(); },
            "Bytes of the largest shard, aligned to L1 alignment.");

    const char* doc = R"doc(
        Allocate a kernel config buffer on a device for a program compiled for it: L1 on the program's Tensix cores,
        each core's shard sized for its own kernel group's Tensix kernel binaries, at a per-core address.

        Pass it to the program's run function to run the binaries from it instead of the KERNEL_CONFIG window, whose
        size then limits only the rest of the kernel configuration: runtime arguments, semaphores, and CB and DFB
        configs. The L1 is freed when the buffer is garbage collected: drop the last reference to free it.

        Needs slow dispatch and the device opened with ``TT_METAL_ALLOCATOR_MODE_HYBRID=1``.

        Args:
            device (ttnn.MeshDevice): The device ``compiled_program`` was compiled for.
            compiled_program (ttnn.experimental.CompiledProgram): The program to size the buffer for.
            bottom_up (bool): Allocate it from the bottom of L1 instead of the top, where the program's static CBs
                would go: its CBs must then be backed by L1 tensors. Defaults to False.

        Returns:
            KernelConfigBuffer: The buffer.
        )doc";
    ttnn::bind_function<"create_kernel_config_buffer", "ttnn.experimental.">(
        mod,
        doc,
        &create_kernel_config_buffer,
        nb::arg("device"),
        nb::arg("compiled_program"),
        nb::arg("bottom_up") = false);
}

}  // namespace ttnn::compiled_program

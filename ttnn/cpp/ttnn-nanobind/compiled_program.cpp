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
    ttnn::device_operation::CompiledProgram& compiled, bool lockstep_address) {
    return ttnn::device_operation::create_kernel_config_buffer(compiled, lockstep_address);
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
            L1 that a compiled program's Tensix kernel-configuration blocks run from, at a per-core address, instead
            of the KERNEL_CONFIG window. Slow dispatch only. Experimental API; may change.
        )doc")
        .def_prop_ro(
            "max_size",
            [](const tt::tt_metal::experimental::KernelConfigBuffer& self) { return self.max_size(); },
            "Bytes of the largest shard, aligned to L1 alignment.")
        .def(
            "deallocate",
            &tt::tt_metal::experimental::KernelConfigBuffer::deallocate,
            "Free the buffer's L1 now. Running with a deallocated buffer is an error.");

    const char* doc = R"doc(
        Allocate a kernel config buffer for a compiled program: L1 on its Tensix cores, each core's shard sized for
        its own kernel group's Tensix kernel-configuration block, at a per-core address.

        Pass it to the program's run function to launch the program from it instead of the KERNEL_CONFIG window,
        whose size then does not limit the program. The L1 is freed by ``deallocate()`` or when the buffer is
        garbage collected.

        Needs slow dispatch and the device opened with ``TT_METAL_ALLOCATOR_MODE_HYBRID=1``. Rejects programs with
        Tensix program semaphores unless ``lockstep_address``.

        Args:
            compiled_program (ttnn.experimental.CompiledProgram): The program to size the buffer for.
            lockstep_address (bool, optional): Put every core's shard at one address, sized for the largest block,
                so the program may use Tensix program semaphores. Needs the cores' free L1 to start at the same
                address. Default: False.

        Returns:
            KernelConfigBuffer: The buffer.
        )doc";
    ttnn::bind_function<"create_kernel_config_buffer", "ttnn.experimental.">(
        mod, doc, &create_kernel_config_buffer, nb::arg("compiled_program"), nb::arg("lockstep_address") = false);
}

}  // namespace ttnn::compiled_program

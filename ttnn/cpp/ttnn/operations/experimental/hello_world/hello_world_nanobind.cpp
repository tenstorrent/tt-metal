// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hello_world_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/hello_world/hello_world.hpp"

namespace ttnn::operations::experimental::hello_world_binding::detail {

void bind_experimental_hello_world_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a new tensor with the same data as the input tensor (an identity op).

        This is an onboarding op: it does nothing to the data, but it exercises the full
        classic TT-Metalium dataflow path (reader -> compute -> writer kernels, circular
        buffers, work split, program cache), and it makes that path observable through
        two independently enableable debug channels:

        Host-side call trace (``log_debug``, category ``Op``):
            Requires a build compiled with ``-DTT_METAL_ENABLE_LOGGING=ON`` (on by default
            for development builds, e.g. ``./build_metal.sh --development``; a default
            Release build compiles the trace out). Then, at runtime:

                TT_LOGGER_LEVEL=debug python your_script.py

            The trace is tagged ``[hello_world]`` and follows the call order:
            launch -> validate -> compute_output_specs -> create_output_tensors ->
            select_program_factory -> create_descriptor (circular buffers, work split,
            kernels, per-core runtime args).

        Device-side per-core DPRINT:
            Works in any build. At runtime:

                TT_METAL_DPRINT_CORES=all python your_script.py

            Each core the workload is placed on prints one line from the compute
            kernel, e.g. ``Hello, world! I am core (0, 0) and I process 1 tile(s).``
            Set ``TT_METAL_DPRINT_FILE=hello_world.log`` to capture the prints in a file.

        Args:
            input_tensor (ttnn.Tensor): input tensor. TILE layout, DRAM interleaved, any dtype.

        Returns:
            ttnn.Tensor: a new tensor with the same shape/dtype/layout/memory config and the same data.

        Example:
            >>> x = ttnn.from_torch(torch.rand(1, 1, 32, 64, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            >>> y = ttnn.experimental.hello_world(x)
    )doc";

    ttnn::bind_function<"hello_world", "ttnn.experimental.">(
        mod, doc, &ttnn::operations::experimental::hello_world, nb::arg("input_tensor").noconvert());
}

}  // namespace ttnn::operations::experimental::hello_world_binding::detail

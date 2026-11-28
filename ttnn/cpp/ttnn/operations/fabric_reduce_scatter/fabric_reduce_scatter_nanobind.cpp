// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fabric_reduce_scatter_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "device/fabric_reduce_scatter_device_operation_types.hpp"
#include "fabric_reduce_scatter.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::fabric_reduce_scatter {

namespace {

// A support refusal reaches Python as the ttnn.operations._op_contract exception of the same name, the
// NotImplementedError subclasses the eval harness recognizes as a deliberate refusal. Any other exception
// falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    }
    // The Python op's asserts (exact_fp32_add needs FLOAT32, columns must split, program_hash needs
    // cache_descriptors, prepared needs program_hash) stay AssertionError.
    catch (const AssertionRefusal& refusal) {
        PyErr_SetString(PyExc_AssertionError, refusal.what());
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

}  // namespace

void bind_fabric_reduce_scatter_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"fabric_reduce_scatter">(
        mod,
        R"doc(
        Reduce-scatter over the fabric, DRAM to DRAM: every chip's input is a partial [1, 1, G * Sb, W] (TILE, DRAM
        interleaved; block i = rows i Sb ..), and every chip gets its block of the sum over its group, [1, 1, Sb, W]
        (the layout ttnn.reduce_scatter(dim=2) gives). Chip (r, c) owns the block of its row-major rank in the group,
        whatever order the group travels in.

        Args:
            input_tensor (ttnn.Tensor): the per-chip partials.

        Keyword Args:
            cluster_axis (int, optional): 0 / 1: a line or ring along that mesh axis; None: one group over the whole
                mesh (a snake). Defaults to 0.
            topology (ttnn.Topology): Linear, or Ring (a group of 2 is a line). Defaults to Linear.
            num_links (int): links per hop. Defaults to 1.
            scheme (str): "ring", or "dual_cycles" (cluster_axis=None, Ring, a torus with both sides >= 3).
                Defaults to "ring".
            output (ttnn.Tensor, optional): a preallocated output [1, 1, Sb, W], written in place.
            placement (str): "auto" (ports next to their link's Ethernet core) or "simple". Defaults to "auto".
            dim (int): 2 = blocks of rows; 3 = blocks of columns (input [1, 1, S, G * Wb] -> [1, 1, S, Wb]).
                Defaults to 2.
            exact_fp32_add (bool): FLOAT32 partials only; every add one IEEE fp32 rounding on the SFPU.
                Defaults to False.
            cache_descriptors (bool): kept for compatibility; the program cache serves every call.
            program_hash (bool): kept for compatibility; needs cache_descriptors.
            prepared (bool): kept for compatibility; needs program_hash.
            scratch_tensor (ttnn.Tensor, optional): caller-held scratch [1, 1, (G + 1) Sb, W]
                (see fabric_reduce_scatter_scratch_tensor); allocated per call when absent.

        Returns:
            ttnn.Tensor: each chip's block of the sum.

        Raises:
            AssertionError: exact_fp32_add without FLOAT32, blocks that do not split into tiles, program_hash
                without cache_descriptors, prepared without program_hash.
            ValueError: a hop that is not a direct link, too few links or cores, an unknown scheme.
        )doc",
        &ttnn::fabric_reduce_scatter,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("cluster_axis") = 0u,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("num_links") = 1u,
        nb::arg("scheme") = "ring",
        nb::arg("output") = nb::none(),
        nb::arg("placement") = "auto",
        nb::arg("dim") = 2,
        nb::arg("exact_fp32_add") = false,
        nb::arg("cache_descriptors") = false,
        nb::arg("program_hash") = false,
        nb::arg("prepared") = false,
        nb::arg("scratch_tensor") = nb::none());

    mod.def(
        "prepare",
        &ttnn::fabric_reduce_scatter_prepare,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cluster_axis") = 0u,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("num_links") = 1u,
        nb::arg("scheme") = "ring",
        nb::arg("placement") = "auto",
        R"doc(The plan and the counters of one group configuration, built once per mesh (the first build creates the
        global semaphores and synchronizes the mesh): call it at load time so that no call pays for it.)doc");

    // The Python view of the program_hash path's descriptor entries (the package's _DESC_CACHE), built only
    // when read: one tuple (prepared, input address, scratch address, output address, programs) per entry.
    mod.def("descriptor_cache_entries", []() {
        nb::list entries;
        for (const auto& entry : ttnn::fabric_reduce_scatter_descriptor_entries()) {
            entries.append(nb::make_tuple(
                entry.prepared, entry.input_address, entry.scratch_address, entry.output_address, entry.num_programs));
        }
        return entries;
    });
}

}  // namespace ttnn::operations::fabric_reduce_scatter

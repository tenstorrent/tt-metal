// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_matmul_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/all_to_all_matmul_device_operation_types.hpp"
#include "all_to_all_matmul.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::all_to_all_matmul {

namespace {

// A support refusal reaches Python as the ttnn.operations._op_contract exception of the same name, the
// NotImplementedError subclasses the eval harness recognizes as a deliberate refusal. Any other exception
// falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const UnsupportedAxisValue& refusal) {
        const nb::object type = nb::module_::import_("ttnn.operations._op_contract").attr("UnsupportedAxisValue");
        PyErr_SetString(type.ptr(), refusal.what());
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

}  // namespace

void bind_all_to_all_matmul_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"all_to_all_matmul">(
        mod,
        R"doc(
        Fused all-to-all (head-sharded -> sequence-sharded) and per-head matmul over a ring.

        Within each group of G devices along ``cluster_axis``, device q ends with rows [q*S/G, (q+1)*S/G) of every
        source's chunk for all H = G*Hl heads, each head multiplied by its own (K, N) matrix of a replicated weight:

            chunks[p] = chunk(x[p], G, dim=2);  a2a[q] = cat_p(chunks[p][q], dim=1);  out[q] = a2a[q] @ w (per head)

        Port cores send every remote quarter over the fabric straight into the L1 of the compute core that
        multiplies it; compute cores read their weights from DRAM while the rows are in flight.

        Args:
            input_tensor (ttnn.Tensor): (B, Hl, S, K) per device, bfloat16, TILE, DRAM interleaved.
            weight (ttnn.Tensor): (1, H, K, N) replicated, bfloat16 or bfloat8_b, TILE, interleaved.

        Keyword Args:
            cluster_axis (int): mesh axis of the ring groups.
            topology (ttnn.Topology): fabric topology. Defaults to ``ttnn.Topology.Ring``.
            num_links (int): links per ring direction. Defaults to 2.
            memory_config (ttnn.MemoryConfig, optional): output placement (interleaved). Defaults to DRAM.
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): defaults to HiFi4, fp32 DEST
                accumulation, no approximation.
            scratch_tensor (ttnn.Tensor, optional): landing scratch from ``ttnn.all_to_all_matmul_scratch_tensor``;
                allocated per call when absent.

        Returns:
            ttnn.Tensor: (B, H, S/G, N) per device, the input's dtype, TILE.

        Raises:
            ValueError: inputs that do not fit together (ranks, heads, K, S % G, links, placement).
            RuntimeError: the fabric payload is smaller than one tile, or the plan does not fit.
            ttnn.operations._op_contract.UnsupportedAxisValue: an input outside the support contract.
        )doc",
        &ttnn::all_to_all_matmul,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("topology") = tt::tt_fabric::Topology::Ring,
        nb::arg("num_links") = 2,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scratch_tensor") = nb::none());

    ttnn::bind_function<"all_to_all_matmul_scratch_tensor">(
        mod,
        R"doc(
        The landing scratch tensor a call of ``ttnn.all_to_all_matmul`` with these arguments uses. Pass it as
        ``scratch_tensor`` to skip the per-call allocation (and keep a captured trace valid while it is held).
        )doc",
        &ttnn::all_to_all_matmul_scratch_tensor,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("topology") = tt::tt_fabric::Topology::Ring,
        nb::arg("num_links") = 2,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::all_to_all_matmul

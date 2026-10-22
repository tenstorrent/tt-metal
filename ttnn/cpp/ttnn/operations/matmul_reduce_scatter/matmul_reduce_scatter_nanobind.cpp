// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/matmul_reduce_scatter_device_operation_types.hpp"
#include "matmul_reduce_scatter.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

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
    } catch (const NotImplementedRefusal& refusal) {
        PyErr_SetString(PyExc_NotImplementedError, refusal.what());
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

}  // namespace

void bind_matmul_reduce_scatter_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"matmul_reduce_scatter">(
        mod,
        R"doc(
        Fused multi-device matmul + reduce-scatter (SUM) over one mesh axis.

        Every device of a group holds a K-slice A[g] (..., M, K) and W[g] (K, N). The op computes each device's
        partial A[g] @ W[g] and reduce-scatters the sum over the group along ``scatter_dim``.

        Args:
            input_tensor (ttnn.Tensor): A, (..., M, K), rank 2-4 with leading dims 1, bfloat16 or bfloat8_b,
                TILE layout, DRAM interleaved.
            weight (ttnn.Tensor): W, (K, N), bfloat16, bfloat8_b or bfloat4_b, TILE layout, DRAM interleaved.

        Keyword Args:
            cluster_axis (int): the mesh axis of the group (0 or 1).
            scatter_dim (int): -1 or -2 (a non-negative value counts from the front). Defaults to -2.
            topology (ttnn.Topology): Linear or Ring. Defaults to Linear.
            num_links (int, optional): fabric links per hop (1 or 2); None uses every usable link.
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): math fidelity / approx mode;
                defaults to HiFi2. fp32 DEST accumulation is always on.
            memory_config (ttnn.MemoryConfig, optional): interleaved output placement; defaults to DRAM.
            output_tensor (ttnn.Tensor, optional): preallocated output matching the op's output spec exactly.

        Returns:
            ttnn.Tensor: the bfloat16 TILE output, the scattered extent divided by the group size.

        Raises:
            UnsupportedAxisValue: an input outside the support contract.
            ValueError: inputs that do not fit together (shapes, placement, group, links, fabric config).
            NotImplementedError: a sharded output memory_config, or a shape that does not fit L1.
        )doc",
        &ttnn::matmul_reduce_scatter,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("scatter_dim") = -2,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::matmul_reduce_scatter

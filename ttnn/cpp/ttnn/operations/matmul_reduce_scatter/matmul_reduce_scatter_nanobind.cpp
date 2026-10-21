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

        Every device of a group along ``cluster_axis`` holds a K-slice A[g] (..., M, K) and W[g] (K, N); the op
        computes each partial A[g] @ W[g] and reduce-scatters the sum along ``scatter_dim``, so device p of the group
        holds block p of sum_g A[g] @ W[g].

        Args:
            input_tensor (ttnn.Tensor): A, (..., M, K) with leading dims 1, DRAM interleaved, TILE, bfloat16 or bfloat8_b.
            weight (ttnn.Tensor): W, (K, N), DRAM interleaved, TILE, bfloat16, bfloat8_b or bfloat4_b.

        Keyword Args:
            cluster_axis (int): the mesh axis (0 or 1) the group runs along.
            scatter_dim (int): -2 (M) or -1 (N). Defaults to -2.
            topology (ttnn.Topology): Linear or Ring. Defaults to Linear.
            num_links (int, optional): fabric links per hop (1 or 2). Defaults to every usable link.
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): math fidelity / approx mode. Defaults to
                HiFi2 with fp32 DEST accumulation.
            memory_config (ttnn.MemoryConfig, optional): interleaved output placement. Defaults to DRAM interleaved.
            output_tensor (ttnn.Tensor, optional): preallocated output matching the spec the op would allocate.

        Returns:
            ttnn.Tensor: bfloat16, TILE, the input's shape with N in the last dim and the scattered dim divided by G.

        Raises:
            UnsupportedAxisValue: an input outside SUPPORTED.
            ValueError: shapes, placements, group size or links that do not fit together.
            NotImplementedError: a sharded output memory_config, or a shape whose plan misses L1.
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

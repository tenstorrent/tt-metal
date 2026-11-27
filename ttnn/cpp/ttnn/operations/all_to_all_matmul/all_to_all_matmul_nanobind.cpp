// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_matmul_nanobind.hpp"

#include <exception>
#include <stdexcept>
#include <tuple>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

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

// One Python entry for both modes: qk_rope_head_dim None -> ttnn.Tensor (the plain op), set -> (prefix, rope_seq).
std::variant<Tensor, std::tuple<Tensor, Tensor>> all_to_all_matmul_any(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology,
    int32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<Tensor>& scratch_tensor,
    const std::optional<int32_t>& qk_rope_head_dim,
    const std::optional<MemoryConfig>& rope_memory_config) {
    if (qk_rope_head_dim.has_value()) {
        return ttnn::all_to_all_matmul_qstem(
            input_tensor,
            weight,
            cluster_axis,
            *qk_rope_head_dim,
            topology,
            num_links,
            memory_config,
            compute_kernel_config,
            scratch_tensor,
            rope_memory_config);
    }
    if (rope_memory_config.has_value()) {
        throw std::invalid_argument("all_to_all_matmul: rope_memory_config needs qk_rope_head_dim");
    }
    return ttnn::all_to_all_matmul(
        input_tensor, weight, cluster_axis, topology, num_links, memory_config, compute_kernel_config, scratch_tensor);
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
            qk_rope_head_dim (int, optional): q-stem mode (the MLA early-Q stem in one call). The input is then the
                unsplit q_b_proj output (1, 1, S, Hl*(K+R)) per device, R = qk_rope_head_dim, K = weight.shape[2]:
                head j's columns [j*(K+R), j*(K+R)+K) go through the exchange + matmul, its R rope columns through
                the same exchange only. DRAM or L1 interleaved input; Ring, S/G a multiple of 32.
            rope_memory_config (ttnn.MemoryConfig, optional): q-stem mode: rope_seq placement (interleaved).
                Defaults to DRAM.

        Returns:
            ttnn.Tensor: (B, H, S/G, N) per device, the input's dtype, TILE. In q-stem mode a tuple
            (prefix (1, H, S/G, N), rope_seq (1, H, S/G, R)).

        Raises:
            ValueError: inputs that do not fit together (ranks, heads, K, S % G, links, placement).
            RuntimeError: the fabric payload is smaller than one tile, or the plan does not fit.
            ttnn.operations._op_contract.UnsupportedAxisValue: an input outside the support contract.
        )doc",
        &all_to_all_matmul_any,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("topology") = tt::tt_fabric::Topology::Ring,
        nb::arg("num_links") = 2,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("scratch_tensor") = nb::none(),
        nb::arg("qk_rope_head_dim") = nb::none(),
        nb::arg("rope_memory_config") = nb::none());

    ttnn::bind_function<"all_to_all_matmul_scratch_tensor">(
        mod,
        R"doc(
        The landing scratch tensor a call of ``ttnn.all_to_all_matmul`` with these arguments uses. Pass it as
        ``scratch_tensor`` to skip the per-call allocation (and keep a captured trace valid while it is held).
        Pass the call's ``qk_rope_head_dim`` for a q-stem call.
        )doc",
        &ttnn::all_to_all_matmul_scratch_tensor,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("topology") = tt::tt_fabric::Topology::Ring,
        nb::arg("num_links") = 2,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("qk_rope_head_dim") = nb::none());
}

}  // namespace ttnn::operations::all_to_all_matmul

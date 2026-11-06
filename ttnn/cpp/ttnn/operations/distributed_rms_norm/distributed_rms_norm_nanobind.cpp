// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "distributed_rms_norm_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "device/distributed_rms_norm_device_operation_types.hpp"
#include "distributed_rms_norm.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::distributed_rms_norm {

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
    } catch (const ExcludedCell& refusal) {
        const nb::object type = nb::module_::import_("ttnn.operations._op_contract").attr("ExcludedCell");
        PyErr_SetString(type.ptr(), refusal.what());
    } catch (const RegimeRefusal& refusal) {
        PyErr_SetString(PyExc_NotImplementedError, refusal.what());
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

}  // namespace

void bind_distributed_rms_norm_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"distributed_rms_norm">(
        mod,
        R"doc(
        RMSNorm over a hidden dim sharded across one mesh axis, fused with the cross-device exchange of the per-row
        sum-of-squares partials over the fabric. Every device p of a group of G devices along `cluster_axis` holds
        x[p] (..., M, W_local) and returns

            out[p] = x[p] * rsqrt(sum_g sum_w x[g]^2 / (G * W_local) + epsilon) * gamma[p]

        Args:
            input_tensor (ttnn.Tensor): tiled mesh tensor, float32 / bfloat16 / bfloat8_b, rank 2 to 4, interleaved,
                HEIGHT_SHARDED or WIDTH_SHARDED.

        Keyword Args:
            cluster_axis (int): the mesh axis (0 or 1) the hidden dim is split over.
            gamma (ttnn.Tensor, optional): per-device scale, TILE (1, 1, 1, W) or ROW_MAJOR sticks
                (1, 1, ceil(W / 32), 32), float32 or bfloat16.
            epsilon (float): added to the mean square. Defaults to 1e-6.
            topology (ttnn.Topology): Linear or Ring. Defaults to Linear.
            num_links (int, optional): fabric links per hop; None uses every usable link.
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): defaults to HiFi4, fp32 DEST, exact math.
            memory_config (ttnn.MemoryConfig, optional): output placement; defaults to DRAM for an interleaved input,
                the input's memory config for a sharded one.
            exchange_mode (str, optional): "direct" (default), "relay" or "mcast" (1D fabrics only). Every mode gives
                the same output.
            output_tensor (ttnn.Tensor, optional): preallocated output matching the spec the op would allocate.
            scratch_tensor (ttnn.Tensor, optional): the exchange landing, from
                ttnn.distributed_rms_norm_scratch_tensor; allocated per call when absent.

        Returns:
            ttnn.Tensor: the normalized tensor, the input's shape, dtype and layout.

        Raises:
            ValueError: invalid arguments (gamma shape, cluster_axis, num_links, Ring without a wrapping fabric,
                exchange_mode).
            UnsupportedAxisValue / ExcludedCell: inputs outside the support contract.
            NotImplementedError: layouts the planner cannot build (mcast on a 2D fabric, L1 fit).
        )doc",
        &ttnn::distributed_rms_norm,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("gamma") = nb::none(),
        nb::arg("epsilon") = 1e-6,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("exchange_mode") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("scratch_tensor") = nb::none());

    ttnn::bind_function<"distributed_rms_norm_scratch_tensor">(
        mod,
        R"doc(
        The exchange landing a distributed_rms_norm call with these arguments uses. Pass it back as `scratch_tensor`
        to skip the per-call allocation; it serves every call whose plan needs a landing of exactly its size.
        )doc",
        &ttnn::distributed_rms_norm_scratch_tensor,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("gamma") = nb::none(),
        nb::arg("epsilon") = 1e-6,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("exchange_mode") = nb::none());
}

}  // namespace ttnn::operations::distributed_rms_norm

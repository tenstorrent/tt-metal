// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/all_gather_rms_norm_device_operation_types.hpp"
#include "all_gather_rms_norm.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::all_gather_rms_norm {

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
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

}  // namespace

void bind_all_gather_rms_norm_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"all_gather_rms_norm">(
        mod,
        R"doc(
        Fused all-gather + RMSNorm / LayerNorm over the fabric, one program per device.

        Every device of a cluster_axis group holds its (..., M, W_local) slice; every member returns the full
        normalized (..., M, G * W_local) rows. Per-row partial stats go around the group once, each chip normalizes
        only its own slice, and the port cores send the normalized slices straight into the peers' output pages.

        Args:
            input_tensor (ttnn.Tensor): TILE, float32 / bfloat16 / bfloat8_b, rank 2-4; DRAM interleaved or
                L1 HEIGHT / WIDTH sharded.

        Keyword Args:
            cluster_axis (int): mesh axis of the gather group (0 or 1).
            gamma (ttnn.Tensor, optional): full-W scale, TILE [1, 1, 1, W] or ROW_MAJOR [1, 1, W/32, 32].
            beta (ttnn.Tensor, optional): full-W shift (layer_norm only), gamma's dtype and layout.
            norm_type (str): "rms_norm" or "layer_norm". Defaults to "rms_norm".
            epsilon (float): defaults to 1e-6.
            topology (ttnn.Topology): Linear or Ring. Defaults to Linear.
            num_links (int, optional): links per hop; defaults to min(usable links, 2).
            compute_kernel_config (optional): defaults to HiFi4, fp32 DEST accumulation, exact math.
            memory_config (ttnn.MemoryConfig, optional): output placement; defaults to DRAM interleaved.
            output_tensor (ttnn.Tensor, optional): preallocated output, written in place and returned.
            scratch_tensor (ttnn.Tensor, optional): caller-held staging (see all_gather_rms_norm_scratch_tensor).

        Returns:
            ttnn.Tensor: the normalized gathered tensor.

        Raises:
            UnsupportedAxisValue / ExcludedCell (ttnn.operations._op_contract): inputs outside the support contract.
            ValueError: inputs that do not fit together.
        )doc",
        &ttnn::all_gather_rms_norm,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("gamma") = nb::none(),
        nb::arg("beta") = nb::none(),
        nb::arg("norm_type") = "rms_norm",
        nb::arg("epsilon") = 1e-6,
        nb::arg("topology") = tt::tt_fabric::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("scratch_tensor") = nb::none());

    ttnn::bind_function<"all_gather_rms_norm_scratch_tensor">(
        mod,
        R"doc(
        The scratch tensor a call of ttnn.all_gather_rms_norm with these arguments uses (the R_U staging of a
        W_local % 32 != 0 slice). Hold it and pass it as scratch_tensor to skip the allocation per call.
        )doc",
        &ttnn::all_gather_rms_norm_scratch_tensor,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("cluster_axis"));
}

}  // namespace ttnn::operations::all_gather_rms_norm

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "toy_scaled_add_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/toy_scaled_add_device_operation_types.hpp"
#include "toy_scaled_add.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::toy_scaled_add {

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
    }
}

}  // namespace

void bind_toy_scaled_add_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"toy_scaled_add">(
        mod,
        R"doc(
        Computes ``a + alpha * (b * gamma)``, ``gamma`` an optional row broadcast down the rows.

        Args:
            a (ttnn.Tensor): tiled (32 x 32), bfloat16 or float32; interleaved, or height-sharded on L1.
            b (ttnn.Tensor): same padded shape and placement as ``a``.

        Keyword Args:
            alpha (float): per-call scalar; changing it reuses the cached program. Defaults to 1.0.
            gamma (ttnn.Tensor, optional): tiled, interleaved, one tile-row as wide as ``a``.
            dtype (ttnn.DataType, optional): output dtype. Defaults to ``a``'s.
            memory_config (ttnn.MemoryConfig, optional): output placement. Defaults to ``a``'s.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): math fidelity, fp32 DEST, ...
            output_tensor (ttnn.Tensor, optional): preallocated output; may be ``a`` (in place).

        Returns:
            ttnn.Tensor: the result.

        Raises:
            ttnn.operations._op_contract.UnsupportedAxisValue: an input outside the supported dtypes, layout,
                tile, rank or placement.
            ttnn.operations._op_contract.ExcludedCell: a height-sharded output in DRAM.
        )doc",
        &ttnn::toy_scaled_add,
        nb::arg("a"),
        nb::arg("b"),
        nb::kw_only(),
        nb::arg("alpha") = 1.0f,
        nb::arg("gamma") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::toy_scaled_add

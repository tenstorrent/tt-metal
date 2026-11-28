// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_fp8_rows_nanobind.hpp"

#include <exception>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/pack_fp8_rows_device_operation_types.hpp"
#include "pack_fp8_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::pack_fp8_rows {

namespace {

// A support refusal reaches Python as the ttnn.operations._op_contract exception of the same name, the
// NotImplementedError subclasses the eval harness recognizes as a deliberate refusal. Any other exception
// falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const PackFp8RowsAssertion& refusal) {
        // The Python op refused with `assert`: the same AssertionError and message.
        PyErr_SetString(PyExc_AssertionError, refusal.what());
    }
}

// A caller's output_tensor is returned as is (the Python op returned the very object it was given): it arrives as the
// pointer to the C++ tensor its Python object holds, and returning that pointer by reference makes nanobind hand back
// the existing Python object instead of wrapping a copy.
Tensor* pack_fp8_rows_into(
    const Tensor& e4m3,
    const Tensor& scale,
    Tensor* output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::CoreCoord>& max_grid) {
    ttnn::pack_fp8_rows(e4m3, scale, *output_tensor, memory_config, max_grid);
    return output_tensor;
}

}  // namespace

void bind_pack_fp8_rows_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"pack_fp8_rows">(
        mod,
        R"doc(
        Packs per-token fp8 rows into 192-byte rows: row m = [e4m3 row m (128 B) | scale m as fp32 (4 B) | 60 zero B].
        A row whose 128 values are all +-0 is written all zero. On a mesh, the same program on every chip.

        Args:
            e4m3 (ttnn.Tensor): [.., M, 128] FP8_E4M3 ROW_MAJOR, DRAM interleaved.
            scale (ttnn.Tensor): [.., M, 1] FLOAT32 ROW_MAJOR, DRAM interleaved, leading dims equal to e4m3's.

        Keyword Args:
            output_tensor (ttnn.Tensor, optional): preallocated uint8 ROW_MAJOR [.., M, 192] DRAM interleaved output,
                returned as is.
            memory_config (ttnn.MemoryConfig, optional): the allocated output's memory config (default DRAM).
            max_grid (ttnn.CoreCoord, optional): the cores are taken from (0, 0) .. (max_grid - 1); default the whole
                compute grid.

        Returns:
            ttnn.Tensor: [.., M, 192] UINT8 ROW_MAJOR.

        Raises:
            AssertionError: an input outside the contract above.
        )doc",
        // tried first: a caller's output_tensor (None does not match a pointer argument without .none())
        ttnn::overload_t(
            &pack_fp8_rows_into,
            nb::arg("e4m3"),
            nb::arg("scale"),
            nb::kw_only(),
            nb::arg("output_tensor"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("max_grid") = nb::none(),
            nb::rv_policy::reference),
        ttnn::overload_t(
            &ttnn::pack_fp8_rows,
            nb::arg("e4m3"),
            nb::arg("scale"),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("max_grid") = nb::none()));
}

}  // namespace ttnn::operations::pack_fp8_rows

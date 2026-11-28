// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "unpack_fp8_rows_nanobind.hpp"

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "device/unpack_fp8_rows_device_operation.hpp"
#include "device/unpack_fp8_rows_device_operation_types.hpp"
#include "unpack_fp8_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::unpack_fp8_rows {

namespace {

// The Python op refused with `assert`: its refusals reach Python as the builtin AssertionError with the same
// message. Any other exception falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const UnpackFp8RowsAssertion& refusal) {
        PyErr_SetString(PyExc_AssertionError, refusal.what());
    }
}

// The programs the op has built, as the keys the Python op's _PREPARED held: ("unpack", mesh id, rows shape,
// out shape, str(rows memory config), str(out memory config), ncores, max_grid or None, common_values). Built
// only when read (fp8_rows._PREPARED), never on a call.
nb::list program_keys() {
    nb::list keys;
    for (const auto& key : unpack_fp8_rows_prepared_keys()) {
        nb::list rows_shape;
        for (const uint32_t d : key.rows_shape) {
            rows_shape.append(d);
        }
        nb::list out_shape;
        for (const uint32_t d : key.out_shape) {
            out_shape.append(d);
        }
        keys.append(nb::make_tuple(
            "unpack",
            key.mesh_id,
            nb::tuple(rows_shape),
            nb::tuple(out_shape),
            key.rows_mc,
            key.out_mc,
            key.ncores,
            nb::make_tuple(key.grid_x, key.grid_y),
            key.common_values));
    }
    return keys;
}

}  // namespace

void bind_unpack_fp8_rows_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"unpack_fp8_rows">(
        mod,
        R"doc(
        The inverse of pack_fp8_rows, onto the bfloat8_b grid: rows [row_lo, row_hi) of batch ``batch`` of ``rows``
        (one 192-byte row a page: [e4m3 x 128 | fp32 scale | ..]) decoded as e4m3 x scale (fp32 on the SFPU),
        rounded to nearest even onto the bfloat8_b grid, and written in place into ``out`` at rows
        [out_row0, out_row0 + row_hi - row_lo). On a mesh, the same program and rows on every chip.

        Args:
            rows (ttnn.Tensor): [B, .., R, 192] UINT8 ROW_MAJOR in DRAM, interleaved or ND-sharded.
            out (ttnn.Tensor): [.., M, 128] BFLOAT8_B TILE, DRAM interleaved; written in place.

        Keyword Args:
            row_lo (int): first row of the range, a multiple of 32.
            row_hi (int): end of the range, row_hi - row_lo a multiple of 32.
            batch (int): the batch of ``rows`` read. Defaults to 0.
            out_row0 (int, optional): first row of ``out`` written, a multiple of 32. Defaults to row_lo.
            max_grid (ttnn.CoreCoord, optional): cut the device's worker grid to this size.
            whole_grid (bool): run every call on the whole grid, so one program serves every row count.
                Defaults to False.
            common_values (bool): the per-call values as common runtime args (one write a program, not a core).
                Defaults to False.

        Returns:
            ttnn.Tensor: ``out``.

        Raises:
            AssertionError: inputs outside the contract (dtype, layout, placement, range, whole tile rows).
        )doc",
        &ttnn::unpack_fp8_rows,
        nb::arg("rows"),
        nb::arg("out"),
        nb::kw_only(),
        nb::arg("row_lo"),
        nb::arg("row_hi"),
        nb::arg("batch") = 0,
        nb::arg("out_row0") = nb::none(),
        nb::arg("max_grid") = nb::none(),
        nb::arg("whole_grid") = false,
        nb::arg("common_values") = false);

    mod.def("unpack_fp8_rows_program_keys", &program_keys, "The structural keys of the programs built so far.");
}

}  // namespace ttnn::operations::unpack_fp8_rows

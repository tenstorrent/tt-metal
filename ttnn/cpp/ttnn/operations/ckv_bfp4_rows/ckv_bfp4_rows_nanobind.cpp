// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ckv_bfp4_rows_nanobind.hpp"

#include <cstddef>
#include <exception>

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/ckv_bfp4_rows_device_operation_types.hpp"
#include "device/ckv_bfp4_rows_program_factory.hpp"
#include "ckv_bfp4_rows.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::ckv_bfp4_rows {

namespace {

// The Python op refused with `assert`: a ContractAssertion reaches Python as AssertionError with the Python's
// message. Any other exception falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const ContractAssertion& refusal) {
        PyErr_SetString(PyExc_AssertionError, refusal.what());
    }
}

}  // namespace

void bind_ckv_bfp4_rows_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"ckv_bfp4_rows">(
        mod,
        R"doc(
        Rows [src_row0, src_row0 + n) of ``src`` -> bfloat4_b row records [row0, row0 + n) of ``rows``, written in
        place, rounded to nearest even by tt-metal's host packer rule. The same records as
        ``ckv_bfp4(tilize(rows of src), rows, row0)``: each core reads its bands' 32 rows, tilizes, rounds and packs
        them. One cached program serves every src_row0, n and row0; on a mesh, the same values on every chip.

        Args:
            src (ttnn.Tensor): bf16 ROW_MAJOR, last dim 512, one row a page (any rank, rows flattened); DRAM,
                interleaved or ND-sharded.
            rows (ttnn.Tensor): [.., R', 320] uint8 ROW_MAJOR record table, written in place.
            src_row0 (int): first source row.
            n (int): rows to convert, a multiple of 32 (0: nothing is launched).
            row0 (int): first record row.

        Keyword Args:
            max_grid (ttnn.CoreCoord or (x, y), optional): the program's cores are the rectangle (0, 0) ..
                (max_grid - 1) of the device grid. Defaults to the whole grid.
            output_tensor (ttnn.Tensor, optional): preallocated output; must be ``rows`` itself.

        Returns:
            None: the records are written into ``rows`` in place.

        Raises:
            AssertionError: on a wrong src / rows dtype, layout or width, n not a multiple of 32, or a source or
                record row range out of bounds.
        )doc",
        &ttnn::ckv_bfp4_rows,
        nb::arg("src"),
        nb::arg("rows"),
        nb::arg("src_row0"),
        nb::arg("n"),
        nb::arg("row0"),
        nb::kw_only(),
        nb::arg("max_grid") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    // Programs built so far (cache misses), read by the package's _ROWS_PREPARED view when a test inspects it.
    mod.def("_programs_built", []() -> std::size_t { return programs_built(); });
}

}  // namespace ttnn::operations::ckv_bfp4_rows

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nanobind.hpp"

#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "argmax_force.hpp"

namespace ttnn::operations::reduction::detail {
void bind_reduction_argmax_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Returns the indices of the maximum values along a dimension.

            The output is UINT32 in ROW_MAJOR layout. When several elements tie for the
            maximum, the smallest index wins.

            Args:
                input_tensor (ttnn.Tensor): the input tensor. Must be on device and interleaved.

            Keyword args:
                dim (int, optional): the dimension to reduce. If ``None``, the maximum is taken
                    over every element, which requires a ROW_MAJOR input. Default: ``None``.
                keepdim (bool, optional): if ``True``, the reduced dimension is kept with size 1.
                    Default: ``False``.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Default: the input tensor's memory configuration.
                output_tensor (ttnn.Tensor, optional): a preallocated tensor to receive the
                    indices. Must be UINT32, ROW_MAJOR, interleaved, on the same device, and
                    shaped like the result. Default: ``None``.
                maxval_tensor (ttnn.Tensor, optional): a preallocated BFLOAT16 ROW_MAJOR tensor
                    shaped like the index output. If given, it receives the maximum value found
                    at each index, so a second pass with :func:`ttnn.max` is not needed to
                    recover it. Only available for the Blackhole cases described below;
                    supplying it for any other call raises an error rather than returning an
                    unwritten tensor. Default: ``None``.
                exact_special_values (bool, optional): controls how special values are
                    compared when reducing the last dimension on Blackhole. By default a faster
                    implementation may run, and on inputs holding NaNs, signed zeros or
                    denormals it can return a *different index*: NaN compares as an infinity of
                    the same sign (so a +inf earlier in the row wins instead of the NaN, and a
                    -NaN never wins), +0 and -0 compare equal (so the earliest zero wins), and
                    denormals compare as zero. Ordinary finite values, including ties, always
                    give identical results. Set this to ``True`` to force the comparison order
                    used by the reference: a sign-magnitude ordering of the bfloat16 bit
                    patterns, smallest index winning ties. It can reduce throughput, but never
                    changes results for ordinary data.

                    This applies to last-dimension reductions only. Reducing a batch or channel
                    dimension always compares as IEEE floating point, on every architecture, and
                    this flag does not change that. Default: ``False``.
                sub_core_grids (ttnn.CoreRangeSet, optional): restricts execution to a subset of
                    cores. Honoured on ROW_MAJOR last-dimension reductions (at most 2 core
                    ranges; more raises), on batch and channel dimension reductions, and for the
                    Blackhole cases described below. Ignored on other calls rather than raising.
                    Default: ``None``.

            Returns:
                ttnn.Tensor: the indices of the maximum values, as UINT32.

            Supported inputs:

            - ``dim=None`` (reduce every element): ROW_MAJOR only; BFLOAT16, FLOAT32, INT32,
              UINT32, UINT16.
            - last dimension: ROW_MAJOR with the same dtypes as above, or TILE with BFLOAT16 or
              FLOAT32.
            - second-to-last dimension: BFLOAT16 or FLOAT32.
            - batch or channel dimension (rank >= 3): BFLOAT16 or FLOAT32.

            Sharded tensors are not supported; inputs and outputs must be interleaved.

            On Blackhole, reducing the last dimension of a BFLOAT16 tensor runs a faster
            implementation when the tensor is in TILE layout, is interleaved, and its last
            dimension is a multiple of 32. This is chosen automatically and does not change
            results, apart from the special values noted under ``exact_special_values``.

            Do not pad the reduction dimension to reach that multiple unless you pad with
            ``-inf``: padding with zero (or any value above the row's true maximum) makes the
            padding win, and the returned index points into it.

            Example:

            .. code-block:: python

                # index of the largest value in each row
                logits = ttnn.from_torch(
                    torch.randn(1, 1, 32, 4096), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)
                indices = ttnn.argmax(logits, dim=-1)

                # keep the reduced dimension
                indices = ttnn.argmax(logits, dim=-1, keepdim=True)

                # index of the largest element in the whole tensor
                flat = ttnn.from_torch(
                    torch.randn(64), dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                index = ttnn.argmax(flat)

        )doc";

    ttnn::bind_function<"argmax">(
        mod,
        doc,
        &ttnn::argmax,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("exact_special_values") = false,
        nb::arg("maxval_tensor") = nb::none());

    // Function-local statics: nanobind is handed a const char* and these must outlive the call.
    static const std::string force_doc_suffix =
        R"doc(
            Args:
                input_tensor (ttnn.Tensor): On-device, INTERLEAVED input.

            Keyword Args:
                dim (int, optional): Dim to reduce.
                keepdim (bool, optional): Keep reduced dim.
                sub_core_grids (CoreRangeSet, optional): Limits execution to a subset of cores.
                memory_config (ttnn.MemoryConfig, optional): Output memory.
                output_tensor (ttnn.Tensor, optional): Preallocated UINT32 ROW_MAJOR output.
                maxval_tensor (ttnn.Tensor, optional): Preallocated BFLOAT16 ROW_MAJOR max-value output;
                    the RVV and SFPU paths only.

            Returns:
                ttnn.Tensor: the UINT32 index output.
        )doc";

    static const std::string force_scalar_reader_doc = std::string(R"doc(
            Verification only: forces the non-vector implementation, so a last-dim reduction runs the
            scalar reader kernels. Note this only bypasses the RVV and SFPU paths -- a batch or channel
            dim reduction still reaches the compute kernel ttnn.argmax would pick for it, so the scalar
            readers are what this runs for the last-dim shapes the suites compare against, not for every
            input. Not part of the ttnn API; use ttnn.argmax, which selects a path on its own. This is
            the reference leg the RVV and SFPU paths are compared against.
        )doc") + force_doc_suffix;

    static const std::string force_rvv_doc = std::string(R"doc(
            Verification only: runs the Blackhole RVV (TRISC2 Zve32f) TILE last-dim path unconditionally,
            raising for a case it cannot serve rather than falling back, so that a comparison against the
            scalar readers cannot silently end up measuring them twice and reporting it as agreement. Not
            part of the ttnn API; use ttnn.argmax, which selects a path on its own.
        )doc") + force_doc_suffix;

    static const std::string force_sfpu_doc = std::string(R"doc(
            Verification only: runs the Blackhole SFPU TILE last-dim path unconditionally, raising for a
            case it cannot serve rather than falling back. Its special-value semantics diverge from the
            scalar readers in the way ttnn.argmax's ``exact_special_values`` documents. Not part of the ttnn
            API; use ttnn.argmax, which selects a path on its own.
        )doc") + force_doc_suffix;

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See argmax_force.hpp.
    mod.def(
        "argmax_force_scalar_reader",
        &argmax_force_scalar_reader,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("maxval_tensor") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        force_scalar_reader_doc.c_str());

    mod.def(
        "argmax_force_rvv",
        &argmax_force_rvv,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("maxval_tensor") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        force_rvv_doc.c_str());

    mod.def(
        "argmax_force_sfpu",
        &argmax_force_sfpu,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("maxval_tensor") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        force_sfpu_doc.c_str());
}

}  // namespace ttnn::operations::reduction::detail

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
            Argmax. Returns indices of maximum values.
            Output is UINT32, ROW_MAJOR, INTERLEAVED (DRAM or L1).

            Args:
                input_tensor (ttnn.Tensor): On-device, INTERLEAVED input.

            Keyword args:
                dim (int, optional): Dim to reduce. ``None`` reduces all elements (ROW_MAJOR input only). Default: ``None``.
                keepdim (bool, optional): Keep reduced dim. Default: ``False``.
                sub_core_grids (CoreRangeSet, optional): Limits execution to a subset of cores. Supported on ROW_MAJOR last-dim reductions (<= 2 ranges), batch/channel dim reductions, and the accelerated last-dim paths described below. Default: ``None``.
                memory_config (ttnn.MemoryConfig, optional): Output memory (INTERLEAVED DRAM/L1). Default: input's memory_config.
                output_tensor (ttnn.Tensor, optional): Preallocated output (must be UINT32, ROW_MAJOR, INTERLEAVED, same device). Default: ``None``.
                exact_special_values (bool, optional): Restricts the automatic path selection below to paths that
                    are bit-identical to the default scalar reader kernels on EVERY input, special values included.
                    Today that admits the scalar readers and the RVV path and excludes the SFPU path, whose compare
                    is IEEE-on-fp32 behind a bf16 special-value gasket rather than those readers' bit-pattern total
                    order -- the same divergence class the existing batch/channel dim compute path ships vs the default
                    scalar reader kernels. Concretely, on the SFPU path (measured on silicon):
                    NaN behaves as same-signed infinity (a NaN row-max yields the NaN's index but max value +inf
                    ``0x7F80``, not the NaN payload; -NaN never wins); -0 flushes to +0 in the max-value output and
                    +0/-0 compare equal (first zero's index is kept); denormals flush to zero before the compare;
                    max values below ~2^-118 carry a +2^-127 additive pack bias. All finite normal data -- including
                    every exact tie -- matches the scalar readers bit-for-bit (smallest index wins ties), so the
                    default is right for ordinary data. Setting this can cost throughput, never correctness.
                    Default: ``False``.
                maxval_tensor (ttnn.Tensor, optional): Preallocated BFLOAT16 ROW_MAJOR tensor with the same logical
                    shape as the index output. Honored only on the accelerated paths: alongside each winning index,
                    it receives the maximum VALUE found at that index, so the caller does not need a second pass over
                    the input with ``ttnn.max`` to recover the value it has just located. Because only those paths
                    can fill it, supplying it on a call that the selection below routes to the scalar readers raises,
                    rather than silently handing back the buffer untouched. Default: ``None``.

            Path selection (automatic):

            The reduction is served by one of three paths and the operation picks one from the input spec; there is
            no argument that names a path. A call is eligible for the two accelerated paths when it runs on a
            Blackhole device over a BFLOAT16 TILE-layout input of rank >= 1 in INTERLEAVED memory, reduces the last
            dim explicitly, uses standard 32x32 tiles, has a last dim that is a multiple of 32, writes an INTERLEAVED
            output, and (if ``output_tensor`` is supplied) that tensor's logical shape is the reduction output shape.
            Everything else runs the scalar reader kernels.

            Among eligible calls the choice keys on the size of the second-to-last dim (H -- the rows per tile-row),
            because that is what the two accelerated paths price differently:

            - **H >= 32** runs on the SFPU (the Tensix vector FPU), multicore, reading TILE layout directly. All 32
              rows of a tile-row are reduced in one lane-parallel pass, so the cost is essentially flat in H; the
              reduction dim is additionally split across cores, with a per-row scalar merge on a gather core. Pass a
              single-core ``sub_core_grids`` to keep it on one core.
            - **H < 32** runs on TRISC2 (the pack-thread RISC-V core) and its Zve32f vector unit ("RVV"), multicore,
              also reading TILE layout directly. Its scan visits every tile once per valid ROW, so it wins wherever
              the SFPU would pay for all 32 lanes to serve fewer than 32 real rows. It splits the reduction dim
              across cores the same way, and merges the per-core candidates in the scalar readers' bit-pattern order,
              so the result is bit-identical however many cores it runs on. It is also the path for eligible calls
              that ask for ``exact_special_values``, at any H.

            A rank-1 input has no second-to-last dim; it counts as H == 1 (one valid row in a tile-row whose other 31
            rows are padding) and always runs on RVV -- it is never routed to the SFPU. Both paths honour any
            ``sub_core_grids``, so the grid never changes which path serves a call. When none is given they pick
            DIFFERENT core counts, because their cost models differ: the SFPU factory uses
            ``ceil(sqrt(1.5 * w_tiles))`` and the RVV factory ``ceil(sqrt(w_tiles * (H + 2)) / 3)``, each capped by
            the compute grid and by ``w_tiles``. Neither takes the whole grid: past those counts both paths scale
            NEGATIVELY, because every additional core adds roughly 0.44 us of per-program dispatch.

            The H = 32 boundary is a measured crossover, not a guess; the measurements behind it and the rationale
            for the exact value live next to ``kSfpuMinRows`` in
            ``ttnn/cpp/ttnn/operations/reduction/argmax/argmax.cpp``.

            Supported:

            - **dim=None** (reduce all elements):

              - input layout: ROW_MAJOR
              - dtypes: BFLOAT16/FLOAT32/INT32/UINT32/UINT16

            - **dim = rank-1** (last / width):

              - ROW_MAJOR input: BFLOAT16/FLOAT32/INT32/UINT32/UINT16 (multi-core by default)
              - TILE input: BFLOAT16/FLOAT32 (single-core scalar readers, or an accelerated path when the call
                qualifies -- see Path selection)

            - **dim = rank-2** (height):

              - BFLOAT16/FLOAT32 only
              - ROW_MAJOR inputs are internally tilized; this path runs single-core

            - **0 <= dim < rank-2** (batch/channel dims, rank >= 3):

              - BFLOAT16/FLOAT32 only (integer dtypes not supported)
              - input may be ROW_MAJOR or TILE (ROW_MAJOR is converted to TILE internally)
              - output is produced in TILE internally and converted to ROW_MAJOR
              - ``sub_core_grids`` is supported (pass a single-core ``CoreRangeSet`` to run on one core)

            Not supported:

            - Sharded tensors (inputs/outputs must be INTERLEAVED)
            - TILE input with ``dim=None``
            - Batch/channel dim reductions with INT/UINT inputs
            - Integer dtypes on batch/channel dim reductions
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
                    accelerated paths only.

            Returns:
                ttnn.Tensor: the UINT32 index output.
        )doc";

    static const std::string force_scalar_reader_doc = std::string(R"doc(
            Verification only: runs the scalar reader kernels unconditionally, on any architecture. Not part
            of the ttnn API; use ttnn.argmax, which selects a path on its own. This is the golden leg the two
            accelerated paths are compared against.
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

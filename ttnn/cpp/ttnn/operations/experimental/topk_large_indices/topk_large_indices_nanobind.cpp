// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "topk_large_indices.hpp"

namespace ttnn::operations::experimental::topk_large_indices::detail {

void bind_topk_large_indices(nb::module_& mod) {
    ttnn::bind_function<"topk_large_indices", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental Top-K over the last dimension of a row-major BFLOAT16 tensor.
        This op is Blackhole-only.

        Returns a ROW_MAJOR UINT32 tensor containing sorted descending top-k indices.
        The output shape matches the input shape except that the last dimension is k.

        This op is intended for large row-major rows. Internally it snaps k to the
        nearest supported LLK size and streams each input row in LLK-sized windows.
        Input values equal to -inf produce the sentinel index 0xFFFFFFFF when they
        survive into the final top-k result.

        K constraints:
            * k must be in [16, 2048];
            * k must be a multiple of 16;
            * the internal LLK window is snapped to 512, 1024, or 2048 elements.

        Input tensor constraints:
            * the input tensor must be allocated on a Blackhole device;
            * rank must be >= 1;
            * all leading dimensions are flattened into independent rows;
            * the flattened leading-dimension row count must fit in uint32_t;
            * the last dimension is the input row length;
            * the flattened row count must be > 0;
            * the last dimension must be >= k and <= 1,073,741,824 elements.

        valid_length (optional):
            * restricts the search to the first ``valid_length`` columns of each row;
            * the remaining columns are ignored -- neither read nor ranked -- so an
              over-allocated row whose tail is stale can be searched without slicing it (a two-level
              program reads whole 64-byte blocks, so stale keys next to the bound are read but never ranked);
            * must be in [1, last dimension]; a bound below k returns sentinels for the missing indices; defaults to the full last dimension;
            * applied at runtime (no recompile), so a loop growing valid_length reuses one program.

        Core selection:
            * ``subdevice_id`` selects a TENSIX subdevice from the active manager;
            * ``sub_core_grids`` optionally restricts work to a fully contained CoreRangeSet;
            * the resolved grid is structural and participates in the program-cache key;
            * kernels and circular buffers are created only on the resolved grid.

        NoC selection:
            * ``dram_read_noc`` picks the NoC the reader streams the input rows from DRAM on;
              the writer (k indices per row, plus the row-split survivor hand-off) takes the other one;
            * defaults to the arch's preferred DRAM-read NoC (NOC_0 on Blackhole);
            * use it when another program runs alongside on a different subdevice and loads one NoC
              heavily, so the two do not contend; it participates in the program-cache key.

        Args:
            input_tensor: device tensor with ROW_MAJOR layout and BFLOAT16 dtype.
            k: required number of indices to return.
            valid_length: optional number of leading columns to search (default: full width).
            valid_length_tensor: optional 1-element UINT32 row-major DRAM tensor read on-device. Use for
                dynamic valid lengths during trace replay. Mutually exclusive with valid_length.
            valid_length_offset: constant added to valid_length_tensor[0] (default 0).
            subdevice_id: optional active subdevice containing the top-k workers.
            sub_core_grids: optional core restriction within ``subdevice_id``.
            valid_end_tensor: optional 1-element UINT32 row-major DRAM tensor holding the REAL token end;
                caps the derived search length at ceil32(valid_end). Requires valid_length_tensor. Without
                it a padded chunk ranks columns that were never written. Pass the SAME bound to
                ring_indexer_score_dsa's valid_end_tensor -- a looser score with a tighter top-k drops real
                keys, the reverse ranks a stale tail.
            dram_read_noc: optional ``ttnn.NOC`` for the reader's DRAM reads (default: arch preferred).
            block_select: optional block size for two-level selection (only 32). The program picks the top 2048 of
                the maxima of 32-key blocks, then the exact top k among the keys of those blocks. The result has the
                same values as single-level selection; which of several tied keys is returned may differ. Applies to
                k = 2048 and a last dimension in [65536, 2097152] that is a multiple of 32; other inputs run the
                single-level program. The program decides at run time, from the search length, whether to run
                two-level. A two-level program takes at most 4 rows per core (unless block_max_tensor is given)
                and, unless the caller supplies block_max_tensor or block_max_output, holds its own
                [rows, ceil(n/32)] BFLOAT16 DRAM maxima tensor for as long as it is cached (a captured trace
                that runs it needs the program kept in the cache).
            two_level_cutover: search length from which a two-level program runs two-level (default: chosen from
                the core count). 0 forces two-level; 0xFFFFFFFF forces single-level for every non-empty row (an
                empty row always runs two-level, which returns sentinels).
            block_max_tensor: optional [rows, ceil(n/32)] BFLOAT16 DRAM tensor of block maxima; skips the program's
                own block-max stage. Only the first ceil(search_length/32) entries of each row are read.
            block_max_output: optional [rows, ceil(n/32)] BFLOAT16 DRAM tensor that receives the computed block
                maxima (testing); not with block_max_tensor.
            mode_output: optional 1-element UINT32 DRAM tensor that receives 1 when the program ran two-level
                (testing).
        )doc",
        &ttnn::experimental::topk_large_indices,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("k"),
        nb::arg("valid_length") = std::nullopt,
        nb::arg("valid_length_tensor") = nb::none(),
        nb::arg("valid_length_offset") = 0,
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("valid_end_tensor") = nb::none(),
        nb::arg("dram_read_noc") = nb::none(),
        nb::arg("block_select") = nb::none(),
        nb::arg("two_level_cutover") = nb::none(),
        nb::arg("block_max_tensor") = nb::none(),
        nb::arg("block_max_output") = nb::none(),
        nb::arg("mode_output") = nb::none());
}

}  // namespace ttnn::operations::experimental::topk_large_indices::detail

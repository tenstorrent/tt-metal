// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_specs.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/metal2_casters.hpp"

#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::program_specs {

namespace m2 = tt::tt_metal;

void py_module_types(nb::module_& mod) {
    // Deriving constructor: build a DataflowBufferSpec from a sharded tensor instead of hand-typing
    // its entry-format fields. The Metal 2.0 counterpart of ttnn.cb_descriptor_from_sharded_tensor.
    mod.def(
        "dfb_spec_from_sharded_tensor",
        &ttnn::dfb_spec_from_sharded_tensor,
        nb::arg("unique_id"),
        nb::arg("tensor"),
        nb::arg("num_entries") = 0,
        nb::arg("borrowed_from") = nb::none(),
        nb::arg("page_as_tile") = false,
        nb::arg("tile_format") = nb::none(),
        nb::arg("advanced_options") = m2::experimental::DFBAdvancedOptions{},
        R"pbdoc(
            Create a DataflowBufferSpec from a sharded tensor.

            Derives every entry-format field from the tensor rather than making you retype what the
            tensor already knows: entry_size (the shard's page stride in L1), num_entries (pages per
            shard), data_format_metadata (from the dtype) and tile_format_metadata (the tensor's tile).
            The Metal 2.0 counterpart of ttnn.cb_descriptor_from_sharded_tensor.

            Args:
                unique_id: The DFB's name within the ProgramSpec.
                tensor: An allocated, sharded tensor to derive the entry format from.
                num_entries: DFB depth in entries. 0 (default) means one entry per page of the
                    shard, i.e. the DFB spans the whole resident shard. Pass a value for a shallower
                    DFB (e.g. a double-buffered staging DFB that only reuses the tensor's format).
                borrowed_from: Name of the TensorParameter whose L1 memory backs this DFB, or None
                    (default) for a normally allocated DFB. A tensor does not know the
                    TensorParameter name you gave it, so this cannot be defaulted for you.
                page_as_tile: Treat entries as tiles even for a ROW_MAJOR tensor - one entry becomes
                    one tile (or the whole shard, when the shard is smaller than a tile) instead of
                    one stick, and tile_format_metadata is set. No-op for a TILE tensor.
                tile_format: Replaces the derived tile_format_metadata. The tile carries the face
                    layout, so pass one when an entry holds fewer or shorter faces than the tensor's
                    tile - which is exactly the sub-tile shard case page_as_tile falls back to. Never
                    derived for you.
                advanced_options: Passed through to advanced_options.

            Returns:
                DataflowBufferSpec

            Raises:
                RuntimeError: if the tensor is not sharded, is not allocated, is asked to back a DFB
                    from non-L1 memory, or the resulting DFB does not fit in the borrowed shard.

            Example:
                >>> # Zero-copy: the DFB *is* the resident shard.
                >>> dfb = ttnn.experimental.dfb_spec_from_sharded_tensor("in", sharded_input, borrowed_from="a")
                >>>
                >>> # Row-major shard read as tiles by a compute kernel.
                >>> dfb = ttnn.experimental.dfb_spec_from_sharded_tensor(
                ...     "in", rm_sharded_input, borrowed_from="a", page_as_tile=True
                ... )
                >>>
                >>> # Separately allocated double-buffered staging DFB, tensor's format only.
                >>> dfb = ttnn.experimental.dfb_spec_from_sharded_tensor("stage", sharded_input, num_entries=2)
        )pbdoc");
}

}  // namespace ttnn::program_specs

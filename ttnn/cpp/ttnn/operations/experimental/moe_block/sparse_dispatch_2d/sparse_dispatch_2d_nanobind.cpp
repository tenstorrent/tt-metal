// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_dispatch_2d_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "sparse_dispatch_2d.hpp"
#include "device/sparse_dispatch_2d_plan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

void bind_sparse_dispatch_2d(nb::module_& mod) {
    ttnn::bind_function<"sparse_dispatch_2d", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
            The MoE block's sparse 2D dispatch: every chip's own rows to the chips that hold one of their experts.

            Each chip (r, c) of the R x C mesh owns S rows, already in place in ``xbuf`` at slots
            ``(r * C + c) * S + t``; ``route`` holds a route word per own row (1 word up to 16 chips: destination mask
            | order << 16 | link << 17; 2 words beyond: the mask, then order | link << 1). Each row travels a
            dimension-ordered tree (X then Y, or Y then X by its order bit) one fabric hop at a time through port cores
            (one per direction and link, next to its Ethernet core), and lands in the same slot of every destination's
            ``xbuf``. Neighbours fence each call, and END markers make the op a barrier: when it returns on a chip,
            every row sent to that chip has landed. Port placement, links and torus wrap come from ``resources``.

            Args:
                xbuf: [1, 1, N * S, W] ROW_MAJOR, DRAM interleaved; filled in place and returned.
                route: [1, 1, 1, RW * S] uint32 ROW_MAJOR, DRAM interleaved: the own rows' route words.
                resources: the block's MoEBlockResources (port cores, semaphore words, plan key).

            Keyword Args:
                group: rows a reader-to-sender hand-off; default 8 when 2 * W <= 7168, else 2.
                manifest_batch: manifest entries a stream sends at once (default 16).
                pack: rows a fabric packet (1 or 2); default 2 when two rows fit the fabric payload.

            Returns:
                xbuf.
        )doc",
        &sparse_dispatch_2d,
        nb::arg("xbuf").noconvert(),
        nb::arg("route").noconvert(),
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("group") = nb::none(),
        nb::arg("manifest_batch") = 16,
        nb::arg("pack") = nb::none());

    mod.def(
        "sparse_dispatch_2d_wiring",
        [](uint32_t R, uint32_t C, bool torus_x, bool torus_y, uint32_t num_links) {
            const auto plan = mesh_plan_uniform(MeshTopology::of(R, C, torus_x, torus_y), num_links);
            nb::dict out;
            for (const auto& [chip, ports] : plan.chips) {
                nb::dict per;
                for (Dir d : kDirs) {
                    if (!ports[static_cast<size_t>(d)]) {
                        continue;
                    }
                    const auto w = wiring(plan, chip, d);
                    nb::list streams;
                    for (const auto& s : w.streams) {
                        streams.append(nb::make_tuple(s.on, std::string(dir_name(s.target)), s.ring));
                    }
                    per[nb::str(std::string(dir_name(d)).c_str())] = nb::make_tuple(w.ring_mask, streams);
                }
                out[nb::make_tuple(chip.r, chip.c)] = per;
            }
            return out;
        },
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("torus_x"),
        nb::arg("torus_y"),
        nb::arg("num_links"),
        R"doc(
            Host-only: each port's wiring on mesh_plan_uniform, {(r, c): {dir: (ring_mask, [(on, target, ring)] x 3)}}
            (sparse_all_gather_2d._wiring).
        )doc");

    mod.def(
        "sparse_dispatch_2d_defaults",
        [](uint32_t width, uint32_t row_bytes, size_t max_payload_bytes) {
            return nb::make_tuple(default_group(width), default_pack(row_bytes, max_payload_bytes));
        },
        nb::arg("width"),
        nb::arg("row_bytes"),
        nb::arg("max_payload_bytes"),
        "Host-only: (group, pack) the op picks for rows of `width` elements, `row_bytes` a page.");
}

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d

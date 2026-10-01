// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pcie_exchange_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pcie_exchange.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::pcie_exchange::detail {

void bind_experimental_pcie_exchange_operation(nb::module_& mod) {
    ttnn::bind_function<"pcie_exchange", "ttnn.experimental.">(
        mod,
        R"doc(
            Two-device collective over Blackhole PCIe peer-to-peer windows (no Ethernet, no host).

            Every core of the L1 width-sharded ``input`` writes its shard into the matching core of the
            peer device through the windows set up by ``ttnn._ttnn.pcie_p2p.setup``; the peer does the same.
            ``exchange`` is a persistent bf16 TILE tensor on the same cores whose shards are at least
            ``2 * ntiles + 1`` tiles wide; it holds the double-buffered landing area and a per-core sequence
            number, so the op is safe to capture into a trace.

            Args:
                input: bf16 TILE L1 width-sharded tensor, one shard per core, same spec on both devices.
                exchange: persistent landing buffer (see above), allocated once through the mesh.
                mode: 0 = all-reduce (sum, output has the input's layout);
                      1 = all-gather along the last dim (output width-sharded over num_devices x N cores,
                      row-major from core (0,0); the input must occupy the first N cores of that walk).

            Keyword Args:
                memory_config: unused for now (the output layout is implied by the mode).
                ctrl: optional persistent control tile per core (bf16 TILE, L1 width-sharded on the input's cores);
                    with it, ``exchange`` only needs the two landing buffers and may be allocated per call.
        )doc",
        &pcie_exchange,
        nb::arg("input").noconvert(),
        nb::arg("exchange").noconvert(),
        nb::arg("mode"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("ctrl") = nb::none(),
        nb::arg("bfp8_payload") = false);
}

}  // namespace ttnn::operations::experimental::pcie_exchange::detail

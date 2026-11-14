// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expert_slot_copy_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "expert_slot_copy.hpp"
#include "device/expert_slot_copy_device_operation_types.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

void bind_expert_slot_copy(nb::module_& mod) {
    ttnn::bind_function<"expert_slot_copy", "ttnn.bringup.moe_block_main.">(
        mod,
        R"doc(
            Table-driven copies of routed-expert weight bytes between chips (2D fabric unicast, any chip to any chip,
            through the port cores of the first hop's direction) or within a chip.

            A run table is one page of ``runs`` (uint32 [1, 1, tables, W] ROW_MAJOR, DRAM interleaved): word 0 the
            number of runs n, then n runs of 6 words - weight (0: gate/up, 1: down), source DRAM bank, source byte
            offset in that bank's shard, destination bank, destination byte offset, bytes. Offsets are relative to the
            tensors' (bank-local) buffer address; each run is cut into fabric-packet chunks.

            Args:
                src_gu, src_d: the weights read (DRAM, sharded over the banks).
                dst_gu, dst_d: the weights written (the same tensors as src_* for a fill between chips).
                runs: the run tables.
                resources: the block's MoEBlockResources (port cores, fabric links, semaphore words, plan key).
                jobs: [(src chip, dst chip, table)], chips r * C + c. src != dst: chip src copies the table's runs into
                    chip dst (a destination at most once a call); the destination's program sends the source a ready
                    when it starts and ends only when every byte has landed, so the copy sits between the programs
                    before it and after it on the destination. src == dst: a local copy (src_* -> dst_*) on that chip.

            Keyword Args:
                links: sender cores (links of the first hop's direction) a fabric copy; default every link of the
                    resources.
                depth, batch: chunk buffers a port core, chunks a DRAM read barrier.

            Returns:
                dst_gu.
        )doc",
        &expert_slot_copy,
        nb::arg("src_gu").noconvert(),
        nb::arg("src_d").noconvert(),
        nb::arg("dst_gu").noconvert(),
        nb::arg("dst_d").noconvert(),
        nb::arg("runs").noconvert(),
        nb::arg("resources"),
        nb::arg("jobs"),
        nb::kw_only(),
        nb::arg("links") = nb::none(),
        nb::arg("depth") = 8,
        nb::arg("batch") = 4);
    mod.def(
        "expert_slot_copy_chunk_bytes",
        &expert_slot_copy_chunk_bytes,
        nb::arg("resources"),
        "The chunk (a fabric packet's payload) expert_slot_copy cuts its runs into, bytes.");
    mod.attr("EXPERT_SLOT_COPY_RUN_WORDS") = kRunWords;
}

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_reduce_scatter — a line / ring reduce-scatter over fabric, DRAM to DRAM.

The operation is the native C++ ttnn.fabric_reduce_scatter (ttnn/cpp/ttnn/operations/fabric_reduce_scatter): it plans
the groups, rings, port / final cores and counters once per mesh, and caches one program per chip.

Input per chip: [1, 1, G * Sb, W] TILE (block j = rows j Sb ..), DRAM interleaved; output per chip: its block of the
sum over the group's G chips, [1, 1, Sb, W] (dim=3: blocks of columns). Chip (r, c) owns the block of its row-major
rank in its group, whatever order the group travels in.

cache_descriptors / program_hash / prepared are kept for their callers: the C++ program cache serves every call, and
the flags only keep their refusals (program_hash needs cache_descriptors, prepared needs program_hash). _DESC_CACHE is
a read-only view of the program_hash calls' entries, built from the C++ op's record when it is read.
"""

from collections.abc import Mapping

import ttnn

_binding = ttnn._ttnn.operations.fabric_reduce_scatter


class _Program:
    """One chip's program of an entry: the C++ op keys every program by its device-operation hash."""

    def __init__(self, custom_program_hash):
        self.custom_program_hash = custom_program_hash


class _Prepared:
    """The prepared path's per-entry state: the entry's (input, scratch, output) tensors."""

    n_io = 3


class _DescriptorCacheView(Mapping):
    """program_hash calls, keyed as the generic op keyed them: ("hashed"[, "prepared"]) + the three addresses."""

    @staticmethod
    def _entries():
        out = {}
        for prepared, in_addr, scr_addr, out_addr, num_programs in _binding.descriptor_cache_entries():
            tag = ("hashed", "prepared") if prepared else ("hashed",)
            key = tag + (in_addr, scr_addr, out_addr)
            programs = [(i, _Program(hash(key) & ((1 << 64) - 1))) for i in range(num_programs)]
            out[key] = (None, programs, _Prepared()) if prepared else (None, programs)
        return out

    def __getitem__(self, key):
        return self._entries()[key]

    def __iter__(self):
        return iter(self._entries())

    def __len__(self):
        return len(self._entries())


_DESC_CACHE = _DescriptorCacheView()

fabric_reduce_scatter = ttnn.fabric_reduce_scatter


def prepare(
    mesh_device, *, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=1, scheme="ring", placement="auto"
):
    """The plan and the counters of one group configuration, built once per mesh by the C++ op (fabric_reduce_scatter
    looks them up on every call). The first build creates the global semaphores and synchronizes the mesh: call it at
    load time so that no forward pass pays for it. The plan stays inside the C++ op; returns mesh_device."""
    _binding.prepare(
        mesh_device,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links,
        scheme=scheme,
        placement=placement,
    )
    return mesh_device

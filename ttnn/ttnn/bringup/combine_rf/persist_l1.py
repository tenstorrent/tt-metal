# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""persist_l1 — where the MoE block's persistent words live (mailboxes, flags, counters, global semaphores).

They outlive a call at addresses other cores' and chips' programs write to, so they cannot be per-call scratch.
In L1 (the default) they sit at the top of the L1 allocator, and the circular-buffer clash check (the lowest occupied
L1 address, device wide) counts them against every other program's static circular buffers: Kimi-K3's chunked KDA
leaves ~9 KB at the top of L1. PERSIST_L1=small puts them in L1_SMALL instead - the region above the L1 allocator
that check does not count (the device must be opened with an l1_small_size that holds them) - and the global
semaphores come from one packed L1_SMALL tensor, 16 B a semaphore (a GlobalSemaphore is a buffer of its own,
64 B a core: the K3 block's ~37 took 2.4 KB)."""

import os

import torch

import ttnn

SEM_BYTES = 16  # (a semaphore a 16 B word: the NOC's write granule - a 4 B write keeps its address's 16 B phase)
SEM_SLOTS = int(os.environ.get("PERSIST_L1_SEMS", "48"))
_POOLS: dict = {}  # id(mesh) -> [mesh, the pool tensor (every worker core), slots handed out]


def small() -> bool:
    return os.environ.get("PERSIST_L1") == "small"


def buffer_type():
    """The buffer type for a persistent word tensor."""
    return ttnn.BufferType.L1_SMALL if small() else ttnn.BufferType.L1


class PackedSemaphore:
    """A global semaphore's stand-in: one 16 B word (zero) of the pool on every worker core of the mesh."""

    def __init__(self, addr: int):
        self.addr = addr


def semaphore(mesh_device, cores, initial_value=0):
    """ttnn.create_global_semaphore, or (PERSIST_L1=small) the pool's next word. Slots are never handed back: a
    block's semaphores live as long as the block."""
    if not small():
        return ttnn.create_global_semaphore(mesh_device, cores, initial_value)
    assert initial_value == 0, "persist_l1: packed semaphores start at zero"
    hit = _POOLS.get(id(mesh_device))
    if hit is None or hit[0] is not mesh_device:
        grid = mesh_device.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        w = SEM_SLOTS * SEM_BYTES // 4
        t = ttnn.from_torch(
            torch.zeros(grid.x * grid.y, w, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1_SMALL,
                ttnn.ShardSpec(crs, (1, w), ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        hit = _POOLS[id(mesh_device)] = [mesh_device, t, 0]
    assert hit[2] < SEM_SLOTS, f"persist_l1: all {SEM_SLOTS} packed semaphores taken (PERSIST_L1_SEMS)"
    s = PackedSemaphore(int(hit[1].buffer_address()) + SEM_BYTES * hit[2])
    hit[2] += 1
    return s


def handoff_bytes() -> int:
    """The fused head's mailbox hand-off words (a word an engine core: the SE reduce-scatter's and the latent's, bumped
    by noc_semaphore_inc, read and zeroed by the port): 16 B apart, or 4 B with PERSIST_L1=small (the mailbox 4176 ->
    ~1.5 KB of L1_SMALL on K3's 110-core engine - atomics need 4 B alignment only)."""
    return 4 if small() else 16


def address(sem) -> int:
    """ttnn.get_global_semaphore_address for either kind."""
    if isinstance(sem, PackedSemaphore):
        return sem.addr
    return int(ttnn.get_global_semaphore_address(sem))

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bank-major expert weight files and a pinned, reusable host staging buffer for the flat routed expert's weights.

Why (measurements: the bring-up notes, /scratch/dnijemcevic/rebase/upload/README.md): the weight cache stores each laid-out weight tensor as a WIDTH_SHARDED DRAM
tensor [1, rows, banks x 32] (one tile column per DRAM bank). In host (tile row-major) order the k-th tile of bank b
sits at tile index k * banks + b, so each bank's pages are 34560 separate 576 B host ranges per chip. tt-metal's
pinned (device-pulls-over-PCIe) write path refuses a sharded buffer whose host ranges are that small
(tt_metal/impl/buffers/dispatch.cpp write_to_device_buffer: batch_large_enough), so every layer pays the serial pin
of the mmapped file (MeshCommandQueue::enqueue_write_tensor try_pin loop) and THEN the slow copy-through-the-issue-
queue path.

The same DRAM bytes as a HEIGHT_SHARDED tensor [1, banks x rows, 32], shard (rows, 32) on the same bank grid: bank b
holds tiles [b T, (b + 1) T) of the host order, T = rows / 32, in the same order as before (the k-th page of bank b
is the k-th tile of what was tile column b). The flat op reads its weights by buffer address and dtype only
(flat_with_program_device_operation.cpp AddrSrc::GateUp / Down; flat_routed_expert.cpp dtype checks), so this tensor
is interchangeable with the width-sharded one. Each bank is ONE contiguous host range: the pinned path applies.

The weight cache's ONLY on-disk layout (owner 2026-11-06: one copy per thing, one layout). File format
(<stem>.bm.bin): a 4096 B header (magic, then JSON: rows, banks, dtype, the width-sharded shape, tile_bytes, mesh,
shard_bytes, stride, n_shards: everything the loader needs, so no other file is read), then one record per mesh
coordinate (row-major), each the chip's shard in bank-major order, at offset 4096 + k * stride (stride = shard bytes
rounded up to 4096, so O_DIRECT / page-aligned reads work). Written (tmp + rename: a file under its final name is
complete) on a cache miss (flat_expert._bank_sharded -> write_from_host). Caches written before 2026-11-06 hold
<stem>.tensorbin (ttnn flatbuffer, WIDTH_SHARDED) instead: never read; the migration is
models/demos/deepseek_v41_flash_d_p/tools/convert_expert_cache.py (host only), after which the .tensorbin can go."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

EXT = ".bm.bin"  # the cache file suffix (flat_expert: <cache_prefix>.flat.<dtype>.lay<key>.<name>.bm.bin)
OLD_EXT = ".tensorbin"  # the pre-2026-11-06 layout of the same stem (convert_expert_cache.py; never read)
PINNED_LIMIT_ENV = "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES"
PINNED_LIMIT_DEFAULT = 4 << 30  # tt-metal's default (rtoptions.cpp) when the variable is unset
MAGIC = b"TTBMAJ01"
HDR = 4096
ALIGN = 4096
CHUNK = 16 << 20  # pread request size


def bm_path(tensorbin) -> str:
    """The bank-major file of an old-layout <stem>.tensorbin (the converter's output name)."""
    s = str(tensorbin)
    assert s.endswith(OLD_EXT), s
    return s[: -len(OLD_EXT)] + EXT


def old_layout_path(bm) -> str:
    """The old-layout <stem>.tensorbin a bank-major file replaces (named in a cache miss's line; never read)."""
    s = str(bm)
    assert s.endswith(EXT), s
    return s[: -len(EXT)] + OLD_EXT


def _round_up(n, a):
    return -(-n // a) * a


# ------------------------------------------------------------------------------------------------ tile permutations
def width_to_bank_major(src: torch.Tensor, banks: int, tile_bytes: int, out: torch.Tensor) -> torch.Tensor:
    """One chip's shard bytes (uint8): host tile (k, b) at k * banks + b  ->  bank-major b * T + k."""
    out.view(banks, -1, tile_bytes).copy_(src.view(-1, banks, tile_bytes).transpose(0, 1))
    return out


def bank_to_width_major(src: torch.Tensor, banks: int, tile_bytes: int, out: torch.Tensor) -> torch.Tensor:
    out.view(-1, banks, tile_bytes).copy_(src.view(banks, -1, tile_bytes).transpose(0, 1))
    return out


# ------------------------------------------------------------------------------------------------ specs
def width_info(t) -> dict:
    """The facts of a width-sharded laid-out weight tensor (host or device) the bank-major copy needs."""
    import ttnn

    mc = t.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and mc.buffer_type == ttnn.BufferType.DRAM, mc
    shape = list(t.shape)
    rows, width = int(shape[-2]), int(shape[-1])
    sh = mc.shard_spec.shape
    assert int(sh[0]) == rows and int(sh[1]) == 32, (sh, shape)
    banks = width // 32
    return {"rows": rows, "banks": banks, "dtype": str(t.dtype).split(".")[-1], "shape": shape}


def hs_memory_config(rows: int, banks: int):
    import ttnn

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (rows, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )


def hs_args(info: dict):
    """(shape, dtype, layout, memory_config) of the bank-major (height-sharded) form of a width-sharded weight."""
    import ttnn

    rows, banks = info["rows"], info["banks"]
    lead = [1] * (len(info["shape"]) - 2)
    return (
        ttnn.Shape(lead + [banks * rows, 32]),
        getattr(ttnn, info["dtype"].lower()),
        ttnn.TILE_LAYOUT,
        hs_memory_config(rows, banks),
    )


def mesh_coords(rows: int, cols: int):
    import ttnn

    return [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]


def shard_views(host_tensor, mesh_shape) -> list[torch.Tensor]:
    """Zero-copy uint8 views of every shard of a multi-device HOST tensor, mesh row-major order."""
    dhb = host_tensor.host_buffer()
    return [torch.from_dlpack(dhb.get_shard(c)) for c in mesh_coords(*mesh_shape)]


# ------------------------------------------------------------------------------------------------ the file
def read_header(path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HDR)
    assert raw[: len(MAGIC)] == MAGIC, f"{path}: not a bank-major weight file"
    return json.loads(raw[len(MAGIC) :].rstrip(b"\0").decode())


def write_bm(path, info: dict, mesh_shape, shards_bank_major) -> None:
    """shards_bank_major: an iterable of uint8 tensors, one per mesh coordinate (row-major), already bank-major."""
    shards = list(shards_bank_major)
    n = mesh_shape[0] * mesh_shape[1]
    assert len(shards) == n, (len(shards), mesh_shape)
    nbytes = int(shards[0].numel())
    hdr = dict(info, mesh=list(mesh_shape), shard_bytes=nbytes, stride=_round_up(nbytes, ALIGN), n_shards=n)
    blob = MAGIC + json.dumps(hdr).encode()
    assert len(blob) <= HDR
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.pwrite(fd, blob.ljust(HDR, b"\0"), 0)
        for k, s in enumerate(shards):
            mv = memoryview(s.numpy())
            off, base = 0, HDR + k * hdr["stride"]
            while off < nbytes:
                off += os.pwrite(fd, mv[off : off + CHUNK], base + off)
        os.ftruncate(fd, HDR + n * hdr["stride"])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def is_complete(path) -> bool:
    """A bank-major file that exists, has a valid header and its full size (the writer renames a finished file into
    place, so this only fails on a foreign or damaged file)."""
    try:
        hdr = read_header(path)
        return os.path.getsize(path) == HDR + hdr["n_shards"] * hdr["stride"]
    except (OSError, ValueError, AssertionError, KeyError):
        return False


def drop_page_cache(path) -> None:
    """Evict a file's clean pages (no root needed): a cold read for a benchmark."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _pread_into(fd, dst: torch.Tensor, file_off: int, lo: int, hi: int) -> None:
    mv = memoryview(dst.numpy())
    off = lo
    while off < hi:
        n = os.preadv(fd, [mv[off : min(hi, off + CHUNK)]], file_off + off)
        if n <= 0:
            raise IOError(f"short read at {file_off + off}")
        off += n


def read_bm_into(path, dsts: list[torch.Tensor], pool: ThreadPoolExecutor, hdr: dict | None = None) -> int:
    """Fill each chip's staging shard (uint8 views) from a bank-major file: parallel preads (the GIL is released in
    the syscall), 16 MB requests, the pool's threads over (shard, 16 MB piece). Returns the bytes read."""
    hdr = hdr or read_header(path)
    n, stride = hdr["shard_bytes"], hdr["stride"]
    assert len(dsts) == hdr["n_shards"] and all(int(d.numel()) == n for d in dsts), (len(dsts), hdr)
    fd = os.open(path, os.O_RDONLY)
    try:
        piece = 4 * CHUNK
        jobs = [(k, lo, min(n, lo + piece)) for k in range(len(dsts)) for lo in range(0, n, piece)]
        list(pool.map(lambda j: _pread_into(fd, dsts[j[0]], HDR + j[0] * stride, j[1], j[2]), jobs))
    finally:
        os.close(fd)
    return n * len(dsts)


# ------------------------------------------------------------------------------------------------ the stager
class PinnedStager:
    """Reusable host staging tensors for the bank-major weights: per (shape, dtype) ``depth`` slots, a slot holding
    one layer's files. The host tensors are allocated once (ttnn.allocate_tensor_on_host: owned anonymous memory,
    never file-backed), so tt-metal's PinnedMemoryCache pins each shard ONCE (keyed by host address) and every later
    copy_host_to_device_tensor is a pure device pull over PCIe (the pinned write path; blocking, so a slot is free
    again when upload returns). ``prefetch`` fills a slot in the background (parallel preads; the GIL is released in
    the syscalls and in copy_host_to_device_tensor). Pinned bytes: depth x one layer (7.6 GB a layer at bfp4); tt-metal
    caches at most TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES (default 4 GiB: less than the 5.1 GB gate/up tensor, so
    some chips would take the issue-queue copy and the pins would be redone every layer): set it to >= depth x 8 GiB."""

    def __init__(self, mesh, depth: int = 2, threads: int = 8):
        self.mesh, self.depth = mesh, int(depth)
        self.shape = tuple(mesh.shape)
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="bm-read")
        self._bg = ThreadPoolExecutor(1, thread_name_prefix="bm-prefetch")  # (drives a fill; the pool does the reads)
        self.sets: dict = {}  # key -> [host tensor] * depth
        self.views: dict = {}  # key -> [[uint8 shard views]] * depth
        self._jobs: dict = {}  # slot -> [set of paths not uploaded yet, Future]
        self._next = 0
        self._lock = threading.Lock()
        self._warned_pin = False

    @staticmethod
    def _key(hdr):
        return (hdr["dtype"], hdr["rows"], hdr["banks"], tuple(hdr["shape"]))

    def staging(self, hdr, slot: int):
        import ttnn

        key = self._key(hdr)
        with self._lock:
            if key not in self.sets:
                shape, dt, lay, mc = hs_args(hdr)
                self.sets[key] = [
                    ttnn.allocate_tensor_on_host(shape, dt, lay, self.mesh, mc) for _ in range(self.depth)
                ]
                self.views[key] = [shard_views(t, self.shape) for t in self.sets[key]]
                self._check_pin_limit()
            return self.sets[key][slot % self.depth], self.views[key][slot % self.depth]

    def staged_bytes(self) -> int:
        """Host bytes of every staging slot (what tt-metal's PinnedMemoryCache must hold to keep them all pinned)."""
        return sum(self.depth * int(v[0][0].numel()) * len(v[0]) for v in self.views.values())

    def _check_pin_limit(self) -> None:
        """One line when tt-metal's pinned-memory cache limit (read when the process first touched tt-metal) is below
        the staging: the upload stays correct (the chips past the limit take the issue-queue copy, still one
        contiguous range a bank), only slower. DeepSeek-V4.1 sets it before the mesh opens (tt/settings.py
        export_pinned_limit)."""
        if self._warned_pin:
            return
        raw = os.environ.get(PINNED_LIMIT_ENV)  # diagnostic: only the log line below reads it
        try:
            limit = PINNED_LIMIT_DEFAULT if raw is None else int(raw, 0)
        except ValueError:
            return
        need = self.staged_bytes()
        if limit < need:
            from loguru import logger

            self._warned_pin = True
            logger.warning(
                f"FLAT_CACHE pinned staging {need / 2**30:.1f} GiB > {PINNED_LIMIT_ENV}="
                f"{raw if raw is not None else f'unset ({PINNED_LIMIT_DEFAULT >> 30} GiB default)'}: some chips take "
                f"the slower issue-queue copy (same bytes); set it to >= {-(-need // 2**30)} GiB before the mesh opens"
            )

    def fill(self, path, slot: int) -> int:
        hdr = read_header(path)
        _, views = self.staging(hdr, slot)
        return read_bm_into(path, views, self.pool, hdr)

    def _free_slot(self):
        """A slot no pending prefetch owns (its files all uploaded, or none), waiting for one if all are owned."""
        for s in range(self.depth):
            job = self._jobs.get(s)
            if job is None or not job[0]:
                return s
        s = min(self._jobs)  # (every slot holds prefetched files not uploaded yet: give up the lowest slot's)
        self._jobs.pop(s)[1].result()
        return s

    def prefetch(self, paths, slot: int | None = None):
        """Start filling a slot (``slot``, or the next free one) from ``paths`` (one layer's files) in the background.
        Returns the Future (its result: the bytes read)."""
        paths = [str(p) for p in paths]
        slot = self._free_slot() if slot is None else slot % self.depth
        prev = self._jobs.get(slot)
        if prev is not None:
            prev[1].result()  # (an earlier fill of this slot must not run under the new one)
        fut = self._bg.submit(lambda: sum(self.fill(p, slot) for p in paths))
        self._jobs[slot] = [set(paths), fut]
        return fut

    def upload(self, path, slot: int | None = None, device_tensor=None):
        """The height-sharded device tensor of one bank-major file: from its prefetched slot (waiting for the fill),
        else read now into ``slot`` / a free slot; then one copy_host_to_device_tensor."""
        import ttnn

        path = str(path)
        if slot is None:
            slot = next((s for s, j in self._jobs.items() if path in j[0]), None)
        else:
            slot %= self.depth
        job = self._jobs.get(slot) if slot is not None else None
        if job is not None and path in job[0]:
            job[1].result()
            job[0].discard(path)
        else:
            if slot is None:
                slot = self._free_slot()
            elif job is not None:
                job[1].result()
                self._jobs.pop(slot)  # (its prefetched files, if any are left, are read again when uploaded)
            self.fill(path, slot)
        hdr = read_header(path)
        assert tuple(hdr["mesh"]) == self.shape, f"{path}: written for mesh {hdr['mesh']}, this mesh is {self.shape}"
        host, _ = self.staging(hdr, slot)
        if device_tensor is None:
            shape, dt, lay, mc = hs_args(hdr)
            device_tensor = ttnn.allocate_tensor_on_device(shape, dt, lay, self.mesh, mc)
        ttnn.copy_host_to_device_tensor(host, device_tensor)  # (releases the GIL; blocking on the pinned path)
        return device_tensor


_STAGERS: dict = {}


def stager(mesh, depth: int | None = None) -> PinnedStager:
    """The process's stager of ``mesh`` (FLAT_CACHE_STAGING_DEPTH slots, default 2)."""
    st = _STAGERS.get(id(mesh))
    if st is None or st.mesh is not mesh:
        d = depth if depth is not None else int(os.environ.get("FLAT_CACHE_STAGING_DEPTH", "2"))
        st = _STAGERS[id(mesh)] = PinnedStager(mesh, depth=d)
    return st


def write_from_host(cache_file, host_tensor) -> str:
    """On a cache miss (flat_expert._bank_sharded): the bank-major file ``cache_file`` (<stem>.bm.bin) of the
    width-sharded, bfp-packed host tensor just laid out (atomic: tmp + rename)."""
    dst = str(cache_file)
    assert dst.endswith(EXT), dst
    info = width_info(host_tensor)
    mesh = tuple(int(x) for x in host_tensor.host_buffer().shape())
    srcs = shard_views(host_tensor, mesh)
    n = int(srcs[0].numel())
    info["tile_bytes"] = n // ((info["rows"] // 32) * info["banks"])
    outs = [width_to_bank_major(s, info["banks"], info["tile_bytes"], torch.empty(n, dtype=torch.uint8)) for s in srcs]
    write_bm(dst, info, mesh, outs)
    return dst

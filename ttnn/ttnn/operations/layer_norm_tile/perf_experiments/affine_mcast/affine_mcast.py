# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""affine_mcast variants for layer_norm_tile's gamma/beta staging.

LN_AFFINE=<variant> selects one variant per pytest process (generic_op caches programs):
  baseline       the op as shipped: every core reads its own gamma/beta copy behind block 0.
  mcast          injector stages once, multicasts both CBs; no receiver handshake, receivers wait
                 where the op barriers today (after block 0's chunks).
  mcast_hs       same, receiver->injector handshake; receivers ack + wait before block 0.
  mcast_early    no handshake, receivers wait before block 0.
  mcast_late     as mcast, but the injector stages behind its own block 0's last chunk.
  mcast_sparse   as mcast, but only the live 64 B rows of each slot are multicast (raw NoC mcast).
  mcast_sparse_late  both.
  mcast_chunk<N>[_late]  whole region as unlinked raw multicasts of N bytes (no path reservation).
  compact[_late] injector reads only the live rows into a compact 3-6 KB block, one linked multicast,
                 every core scatters the rows into its CB slots locally (no handshake).
LN_AFFINE_SENDER=last|first|mid picks the injector from the row assignment (default last: a 1-row core).
"""

import os
from pathlib import Path

import ttnn

from ttnn.operations.layer_norm_tile import layer_norm_tile as _op
from ttnn.operations.layer_norm_tile import layer_norm_tile_program_descriptor as _pd

KERNEL = Path(__file__).parent / "kernels" / "reader_affine_mcast.cpp"
COMPACT_KERNEL = Path(__file__).parent / "kernels" / "reader_affine_mcast_compact.cpp"
CB_AFFINE_SCRATCH = 9

# variant -> (handshake, receive_early, send_late, sparse, chunk_bytes)
VARIANTS = {
    "baseline": None,
    "mcast": (False, False, False, False, 0),
    "mcast_late": (False, False, True, False, 0),
    "mcast_hs": (True, True, False, False, 0),
    "mcast_early": (False, True, False, False, 0),
    "mcast_sparse": (False, False, False, True, 0),
    "mcast_sparse_late": (False, False, True, True, 0),
    "mcast_chunk2k": (False, False, False, False, 2048),
    "mcast_chunk2k_late": (False, False, True, False, 2048),
    "mcast_chunk8k": (False, False, False, False, 8192),
    "mcast_chunk8k_late": (False, False, True, False, 8192),
}
MAX_RECTS = 3

_orig_create = _pd.create_program_descriptor


def _pick_sender(assignment, where):
    if where == "first":
        return assignment[0][0]
    if where == "mid":
        return assignment[len(assignment) // 2][0]
    return assignment[-1][0]


def _rect_args(device, cores, sender):
    """(count, sx, sy, ex, ey, num_dests per rectangle) for the raw sparse multicast; padded to MAX_RECTS."""
    rects = []
    for rng in cores.ranges():
        size = (rng.end.x - rng.start.x + 1) * (rng.end.y - rng.start.y + 1)
        inside = rng.start.x <= sender.x <= rng.end.x and rng.start.y <= sender.y <= rng.end.y
        num_dests = size - int(inside)
        if num_dests == 0:
            continue
        start = device.worker_core_from_logical_core(rng.start)
        end = device.worker_core_from_logical_core(rng.end)
        rects.append([min(start.x, end.x), min(start.y, end.y), max(start.x, end.x), max(start.y, end.y), num_dests])
    assert len(rects) <= MAX_RECTS, rects
    flat = [len(rects)] + [w for r in rects for w in r]
    return flat + [0] * (1 + 5 * MAX_RECTS - len(flat))


def _append_rt(runtime_args, assignment, extra):
    out = ttnn.RuntimeArgs()
    for core, _, _ in assignment:
        out[core.x][core.y] = list(runtime_args[core.x][core.y]) + extra
    return out


def _make_create(handshake, receive_early, send_late, sparse, chunk_bytes, sender_where):
    def create(input_tensor, gamma, beta, residual, output_tensor, blocking, **kwargs):
        base = _orig_create(input_tensor, gamma, beta, residual, output_tensor, blocking, **kwargs)
        kernels = list(base.kernels)
        reader = kernels[0]
        reader.kernel_source = str(KERNEL)
        reader.named_compile_time_args = list(reader.named_compile_time_args) + [
            ("affine_recv_early", int(receive_early)),
            ("affine_send_late", int(send_late)),
            ("affine_sparse", int(sparse)),
            ("affine_chunk_bytes", chunk_bytes),
        ]
        device = input_tensor.device()
        cores = blocking.all_cores
        sender = _pick_sender(blocking.assignment, sender_where)
        rect_args = _rect_args(device, cores, sender)
        reader.runtime_args = _append_rt(reader.runtime_args, blocking.assignment, rect_args)
        descriptor = ttnn.ProgramDescriptor(cbs=list(base.cbs))
        next_sem = 0
        for prefix in ("gmc", "bmc"):
            mc = ttnn.Mcast(
                device,
                ttnn.McastConfig(handshake=handshake),
                cores,
                cores.num_cores(),
                ttnn.McastExplicitFixedSenderConfig([sender]),
            )
            mc.attach(descriptor, prefix, [reader], next_sem)
            next_sem = mc.next_semaphore_id()
        descriptor.kernels = kernels
        return descriptor

    return create


def _make_create_compact(send_late, sender_where):
    def create(input_tensor, gamma, beta, residual, output_tensor, blocking, **kwargs):
        base = _orig_create(input_tensor, gamma, beta, residual, output_tensor, blocking, **kwargs)
        kernels = list(base.kernels)
        reader = kernels[0]
        reader.kernel_source = str(COMPACT_KERNEL)
        reader.named_compile_time_args = list(reader.named_compile_time_args) + [
            ("affine_send_late", int(send_late)),
            ("affine_scratch_cb", CB_AFFINE_SCRATCH),
        ]
        cores = blocking.all_cores
        # Compact block: 2 tensors x Wt entries of 128 B (TILE) or 64 B (ROW_MAJOR).
        entry_bytes = 128 if gamma.layout == ttnn.TILE_LAYOUT else 64
        scratch_bytes = 2 * blocking.tensor_w_tiles * entry_bytes
        scratch = ttnn.CBDescriptor(
            total_size=scratch_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_AFFINE_SCRATCH, data_format=ttnn.bfloat16, page_size=scratch_bytes
                )
            ],
        )
        descriptor = ttnn.ProgramDescriptor(cbs=list(base.cbs) + [scratch])
        mc = ttnn.Mcast(
            input_tensor.device(),
            ttnn.McastConfig(handshake=False),
            cores,
            cores.num_cores(),
            ttnn.McastExplicitFixedSenderConfig([_pick_sender(blocking.assignment, sender_where)]),
        )
        mc.attach(descriptor, "amc", [reader], 0)
        descriptor.kernels = kernels
        return descriptor

    return create


COMPACT_VARIANTS = {"compact": False, "compact_late": True}


def _install_graduated():
    """The graduated patch applied to a private copy of the op (graduated/): descriptor + kernels."""
    import importlib.util
    import sys

    path = Path(__file__).parent / "graduated" / "layer_norm_tile_program_descriptor.py"
    spec = importlib.util.spec_from_file_location("ln_affine_graduated_pd", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass resolution needs the module registered
    spec.loader.exec_module(module)
    _op.create_program_descriptor = module.create_program_descriptor


def install():
    name = os.environ.get("LN_AFFINE", "baseline")
    sender_where = os.environ.get("LN_AFFINE_SENDER", "last")
    if name == "graduated":
        _install_graduated()
        return name
    if name in COMPACT_VARIANTS:
        _op.create_program_descriptor = _make_create_compact(COMPACT_VARIANTS[name], sender_where)
        return name
    cfg = VARIANTS[name]
    if cfg is not None:
        _op.create_program_descriptor = _make_create(*cfg, sender_where)
    return name

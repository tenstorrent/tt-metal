# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""generic_op through a prepared program (ttnn.bringup.prepare_program / run_prepared, prepared_program.hpp).

generic_op(io_tensors, mesh_program_descriptor) re-applies every runtime arg of every core of every chip from the
descriptor on each program-cache hit. A call site that keeps its descriptor object across calls (rebuilding it only
when its structure or the addresses it embeds outside io_tensors change) can call this generic_op instead: the
descriptor is prepared the first time this object is seen, and each call patches only the runtime args holding an
io tensor's address and the CBs an io tensor backs. The contract is prepared_program.hpp's: every per-call tensor a
kernel addresses is in io_tensors, at the same positions; a descriptor object is not mutated after its first call."""

import ttnn

_CAP = 512  # (prepared descriptors kept: a call site that builds a new descriptor every call only cycles this)
_prepared = {}  # id(descriptor) -> (descriptor, prepared): the descriptor kept alive so its id is not reused


def generic_op(io_tensors, mesh_program_descriptor):
    """ttnn.generic_op(io_tensors, mesh_program_descriptor), dispatched as a prepared program."""
    entry = _prepared.get(id(mesh_program_descriptor))
    if entry is None or entry[0] is not mesh_program_descriptor:
        if len(_prepared) >= _CAP:
            _prepared.pop(next(iter(_prepared)))
        entry = (mesh_program_descriptor, ttnn.bringup.prepare_program(mesh_program_descriptor, io_tensors))
        _prepared[id(mesh_program_descriptor)] = entry
    return ttnn.bringup.run_prepared(entry[1], io_tensors)


def clear():
    _prepared.clear()

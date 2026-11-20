# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from . import layout as L
from .program_descriptor import create_program_descriptor

_programs = {}
_startup_semaphores = {}


def _allocate_dram(device, rows, columns):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, columns]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def _startup_addresses(mesh):
    if mesh not in _startup_semaphores:
        relays = L.core_range_set(L.RELAY_CORE_LIST)
        _startup_semaphores[mesh] = [ttnn.create_global_semaphore(mesh, relays, 0) for _ in range(3)]
    return [ttnn.get_global_semaphore_address(semaphore) for semaphore in _startup_semaphores[mesh]]


def _program(tensors, groups):
    device = tensors[0].device()
    key = (device, groups, *((tuple(tensor.shape), tensor.buffer_address()) for tensor in tensors))
    if key not in _programs:
        if not isinstance(device, ttnn.MeshDevice):
            _programs[key] = create_program_descriptor(*tensors, 0, 1, groups, None, None)
        else:
            chips = device.shape[1]
            node_ids = [device.get_fabric_node_id(ttnn.MeshCoordinate(0, chip)) for chip in range(chips)]
            program = ttnn.MeshProgramDescriptor()
            for chip in range(chips):
                coordinate = ttnn.MeshCoordinate(0, chip)
                program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = create_program_descriptor(
                    *tensors, chip, chips, groups, node_ids, _startup_addresses(device)
                )
            _programs[key] = program
    return _programs[key]


def qwen_mlp_prefill(h, w_norm, w_gate_up, w_down, groups=L.MAX_GROUPS):
    y = _allocate_dram(h.device(), *h.shape)
    tensors = [h, w_norm, w_gate_up, w_down, y]
    ttnn.generic_op(tensors, _program(tensors, groups))
    return y

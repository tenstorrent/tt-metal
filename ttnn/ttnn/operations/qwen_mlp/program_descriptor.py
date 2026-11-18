# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
EPS = 1e-6
GRID_X = 11
GRID_Y = 9
FIRST_Y = 1

RING_BLOCKS = 2
NORM_GROUP = 4
NORM_CHUNK = 16

BF16_TILE_BYTES = 2048
BFP4_TILE_BYTES = 576
FP32_TILE_BYTES = 4096


@dataclass(frozen=True)
class MatmulBlocks:
    M: int
    K: int
    N: int
    sub_h: int
    sub_w: int


@dataclass(frozen=True)
class BlockConfig:
    gate_up: MatmulBlocks
    down: MatmulBlocks


BLOCK_CONFIGS = {
    4096: BlockConfig(MatmulBlocks(4, 8, 16, 2, 2), MatmulBlocks(4, 8, 9, 4, 1)),
    2048: BlockConfig(MatmulBlocks(2, 8, 16, 2, 2), MatmulBlocks(2, 8, 9, 2, 1)),
    1024: BlockConfig(MatmulBlocks(3, 8, 16, 1, 4), MatmulBlocks(3, 8, 9, 1, 3)),
}

CB_IN0 = 0
CB_GATE_UP_IN1 = 1
CB_GATE_UP_OUT = 2
CB_GATE_UP_PARTIAL = 3
CB_NORM_IN = 4
CB_NORM_WEIGHT = 5
CB_NORM_SCALER = 6
CB_NORM_SQUARES = 7
CB_NORM_SUM = 8
CB_NORM_RSTD = 9
CB_NORM_SCALED = 10
CB_DOWN_IN1 = 11
CB_DOWN_OUT = 12
CB_DOWN_PARTIAL = 13
CB_NORM_OUT = 14

SEMAPHORES = [
    "in0_sender",
    "in0_receiver",
    "in0_valid",
    "in1_sender",
    "in1_receiver",
    "in1_valid",
    "barrier_norm",
    "barrier_gate_up",
]

TENSORS = ["h", "w_norm", "w_gate_up", "w_down", "n", "a", "y"]


@dataclass(frozen=True)
class MatmulPhase:
    name: str
    blocks: MatmulBlocks
    M_core: int
    N_core: int
    K_tiles: int
    in0_row_tiles: int
    in1_row_tiles: int
    out_row_tiles: int
    out_N_logical: int
    swiglu: bool
    cb_in1: int
    cb_out: int
    cb_partial: int

    @property
    def K_blocks(self):
        return self.K_tiles // self.blocks.K

    def named_args(self, M_logical):
        b = self.blocks
        return {
            "cb_in1": self.cb_in1,
            "cb_out": self.cb_out,
            "cb_partial": self.cb_partial,
            "M_block": b.M,
            "K_block": b.K,
            "N_block": b.N,
            "sub_h": b.sub_h,
            "sub_w": b.sub_w,
            "M_blocks": self.M_core // b.M,
            "N_blocks": self.N_core // b.N,
            "K_blocks": self.K_blocks,
            "in0_row_tiles": self.in0_row_tiles,
            "in1_row_tiles": self.in1_row_tiles,
            "out_row_tiles": self.out_row_tiles,
            "M_logical": M_logical,
            "N_logical": self.in1_row_tiles,
            "out_N_logical": self.out_N_logical,
            "swiglu": int(self.swiglu),
        }

    @property
    def in0_block_tiles(self):
        return self.blocks.M * self.blocks.K

    @property
    def in1_block_tiles(self):
        return self.blocks.K * self.blocks.N

    @property
    def out_block_tiles(self):
        return self.blocks.M * self.blocks.N // (2 if self.swiglu else 1)

    @property
    def partial_block_tiles(self):
        return self.blocks.M * self.blocks.N


def _cb(index, data_format, page_bytes, tiles, cores):
    return ttnn.CBDescriptor(
        total_size=tiles * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_bytes)],
    )


def _chain(length, increasing):
    order = list(range(length)) if increasing else [0] + list(range(length - 1, 0, -1))
    return {
        position: {
            "prev": order[max(index - 1, 0)],
            "dest": order[min(index + 1, length - 1)],
            "is_injector": int(index == 0),
            "is_sink": int(index == length - 1),
        }
        for index, position in enumerate(order)
    }


def create_program_descriptor(h, w_norm, w_gate_up, w_down, n, a, y):
    device = h.device()
    S, H = h.shape
    I_tp = w_down.shape[0]
    M_tiles = S // TILE
    H_tiles = H // TILE
    I_tiles = I_tp // TILE
    config = BLOCK_CONFIGS[S]

    M_core = ceil(M_tiles / GRID_X)
    gate_up = MatmulPhase(
        "mm1",
        config.gate_up,
        M_core,
        2 * ceil(I_tiles / GRID_Y),
        H_tiles,
        H_tiles,
        2 * I_tiles,
        I_tiles,
        I_tiles,
        True,
        CB_GATE_UP_IN1,
        CB_GATE_UP_OUT,
        CB_GATE_UP_PARTIAL,
    )
    down = MatmulPhase(
        "mm2",
        config.down,
        M_core,
        ceil(H_tiles / GRID_Y),
        I_tiles,
        I_tiles,
        H_tiles,
        H_tiles,
        H_tiles,
        False,
        CB_DOWN_IN1,
        CB_DOWN_OUT,
        CB_DOWN_PARTIAL,
    )
    phases = [gate_up, down]
    for phase in phases:
        assert phase.M_core % phase.blocks.M == 0
        assert phase.N_core % phase.blocks.N == 0
        assert phase.K_tiles % phase.blocks.K == 0
        assert phase.blocks.M % phase.blocks.sub_h == 0 and phase.blocks.N % phase.blocks.sub_w == 0
    assert gate_up.in0_block_tiles == down.in0_block_tiles
    assert H_tiles % NORM_CHUNK == 0

    cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, FIRST_Y), ttnn.CoreCoord(GRID_X - 1, FIRST_Y + GRID_Y - 1))]
    )
    bf16, bfp4, fp32 = ttnn.bfloat16, ttnn.bfloat4_b, ttnn.float32
    cbs = [
        _cb(CB_IN0, bf16, BF16_TILE_BYTES, RING_BLOCKS * gate_up.in0_block_tiles, cores),
        _cb(CB_GATE_UP_IN1, bfp4, BFP4_TILE_BYTES, RING_BLOCKS * gate_up.in1_block_tiles, cores),
        _cb(CB_GATE_UP_OUT, bf16, BF16_TILE_BYTES, 2 * gate_up.out_block_tiles, cores),
        _cb(CB_GATE_UP_PARTIAL, fp32, FP32_TILE_BYTES, gate_up.partial_block_tiles, cores),
        _cb(CB_DOWN_IN1, bfp4, BFP4_TILE_BYTES, RING_BLOCKS * down.in1_block_tiles, cores),
        _cb(CB_DOWN_OUT, bf16, BF16_TILE_BYTES, 2 * down.out_block_tiles, cores),
        _cb(CB_DOWN_PARTIAL, fp32, FP32_TILE_BYTES, down.partial_block_tiles, cores),
        _cb(CB_NORM_IN, bf16, BF16_TILE_BYTES, 2 * NORM_CHUNK, cores),
        _cb(CB_NORM_WEIGHT, bf16, BF16_TILE_BYTES, 2 * NORM_CHUNK, cores),
        _cb(CB_NORM_SCALER, bf16, BF16_TILE_BYTES, 1, cores),
        _cb(CB_NORM_SQUARES, bf16, BF16_TILE_BYTES, 2 * NORM_GROUP, cores),
        _cb(CB_NORM_SUM, fp32, FP32_TILE_BYTES, 1, cores),
        _cb(CB_NORM_RSTD, bf16, BF16_TILE_BYTES, 1, cores),
        _cb(CB_NORM_SCALED, bf16, BF16_TILE_BYTES, 2 * NORM_GROUP, cores),
        _cb(CB_NORM_OUT, bf16, BF16_TILE_BYTES, 2 * NORM_GROUP, cores),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(len(SEMAPHORES))]

    tensors = dict(zip(TENSORS, [h, w_norm, w_gate_up, w_down, n, a, y]))
    tensor_accessor_args = []
    tensor_accessor_offsets = {}
    for name, tensor in tensors.items():
        tensor_accessor_offsets[name] = len(tensor_accessor_args)
        tensor_accessor_args += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()

    named = {
        "cb_in0": CB_IN0,
        "cb_norm_in": CB_NORM_IN,
        "cb_norm_weight": CB_NORM_WEIGHT,
        "cb_norm_scaler": CB_NORM_SCALER,
        "cb_norm_squares": CB_NORM_SQUARES,
        "cb_norm_sum": CB_NORM_SUM,
        "cb_norm_rstd": CB_NORM_RSTD,
        "cb_norm_scaled": CB_NORM_SCALED,
        "cb_norm_out": CB_NORM_OUT,
        "norm_group": NORM_GROUP,
        "norm_chunk": NORM_CHUNK,
        "norm_row_stride": GRID_Y,
        "norm_inv_width_bits": struct.unpack("I", struct.pack("f", 1.0 / H))[0],
        "norm_eps_bits": struct.unpack("I", struct.pack("f", EPS))[0],
        "tile_bytes_bf16": BF16_TILE_BYTES,
        "tile_bytes_bfp4": BFP4_TILE_BYTES,
        "barrier_group_size": GRID_Y,
    }
    named.update({f"sem_{name}": i for i, name in enumerate(SEMAPHORES)})
    named.update({f"ta_{name}": offset for name, offset in tensor_accessor_offsets.items()})
    for phase in phases:
        named.update({f"{phase.name}_{key}": value for key, value in phase.named_args(M_tiles).items()})

    in0_chain = _chain(GRID_Y, increasing=True)
    in1_chain = _chain(GRID_X, increasing=False)

    def physical(x, y):
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return core.x, core.y

    runtime_args = ttnn.RuntimeArgs()
    runtime_arg_names = None
    for x in range(GRID_X):
        for row in range(GRID_Y):
            core_y = FIRST_Y + row
            M_start = x * M_core
            norm_first_row = M_start + row
            norm_rows = len(range(norm_first_row, min(M_start + M_core, M_tiles), GRID_Y))
            args = {
                "h_addr": h.buffer_address(),
                "w_norm_addr": w_norm.buffer_address(),
                "w_gate_up_addr": w_gate_up.buffer_address(),
                "w_down_addr": w_down.buffer_address(),
                "n_addr": n.buffer_address(),
                "a_addr": a.buffer_address(),
                "y_addr": y.buffer_address(),
                "norm_first_row": norm_first_row,
                "norm_rows": norm_rows,
                "defer_k_block": 1 + (x * GRID_Y + row) % (min(phase.K_blocks for phase in phases) - 1),
            }
            for name, chain, core_at in [
                ("in0", in0_chain[row], lambda position: (x, FIRST_Y + position)),
                ("in1", in1_chain[x], lambda position: (position, core_y)),
            ]:
                args[f"{name}_prev_x"], args[f"{name}_prev_y"] = physical(*core_at(chain["prev"]))
                args[f"{name}_dest_x"], args[f"{name}_dest_y"] = physical(*core_at(chain["dest"]))
                args[f"{name}_is_injector"] = chain["is_injector"]
                args[f"{name}_is_sink"] = chain["is_sink"]
            args["barrier_x"], args["barrier_y"] = physical(x, FIRST_Y)
            for phase in phases:
                args[f"{phase.name}_M_start"] = M_start
                args[f"{phase.name}_N_start"] = row * phase.N_core
            if runtime_arg_names is None:
                runtime_arg_names = list(args)
            runtime_args[x][core_y] = list(args.values())
    named.update({f"rt_{name}": index for index, name in enumerate(runtime_arg_names)})
    named_compile_time_args = list(named.items())

    def kernel(source, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cores,
            compile_time_args=tensor_accessor_args,
            named_compile_time_args=named_compile_time_args,
            runtime_args=runtime_args,
            config=config,
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(
                "dm_in0.cpp",
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            ),
            kernel(
                "dm_in1.cpp",
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            ),
            kernel(
                "compute.cpp",
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True
                ),
            ),
        ],
        semaphores=semaphores,
        cbs=cbs,
    )

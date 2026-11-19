# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct
from pathlib import Path

import ttnn
from . import layout as L

KERNEL_DIR = Path(__file__).parent / "kernels"
EPS = 1e-6
TENSORS = ["h", "w_norm", "w_gate_up", "w_down", "y"]


def _cb(index, data_format, page_bytes, tiles, cores):
    return ttnn.CBDescriptor(
        total_size=tiles * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_bytes)],
    )


def _role_cbs(chips):
    bf16, bfp4, fp32, act = ttnn.bfloat16, ttnn.bfloat4_b, ttnn.float32, ttnn.bfloat8_b
    t, c = L.HIDDEN_TILES, L.NORM_CHUNK_TILES
    norm, a, b = L.NORM_GRID.core_range_set(), L.a_active_cores(), L.B_GRID.core_range_set()
    relay = L.RELAY_GRID.core_range_set()
    seg = L.B_TILES_PER_CORE
    return [
        _cb(L.CB_NORM_SCALER, bf16, L.BF16_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_STARTUP, bfp4, L.BFP4_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_SQUARES, bf16, L.BF16_TILE_BYTES, c, norm),
        _cb(L.CB_NORM_SUM, fp32, 4096, 1, norm),
        _cb(L.CB_NORM_RSTD, bf16, L.BF16_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_SCALED, bf16, L.BF16_TILE_BYTES, c, norm),
        _cb(L.CB_NORM_WEIGHT, bf16, L.BF16_TILE_BYTES, t, norm),
        _cb(L.CB_NORM_IN, bf16, L.BF16_TILE_BYTES, 2 * t, norm),
        _cb(L.CB_NORM_OUT, act, L.ACT_TILE_BYTES, 2 * t, norm),
        _cb(L.CB_N, act, L.ACT_TILE_BYTES, L.N_DEPTH * t, a),
        _cb(L.CB_WG, bfp4, L.BFP4_TILE_BYTES, t * 2 * L.A_TILES_PER_CORE, a),
        _cb(L.CB_A_OUT, act, L.ACT_TILE_BYTES, 4 * L.A_TILES_PER_CORE, a),
        _cb(L.CB_A, act, L.ACT_TILE_BYTES, 2 * L.K_TILES_PER_B, b),
        _cb(L.CB_WD, bfp4, L.BFP4_TILE_BYTES, L.K_TILES_PER_B * L.N_TILES_PER_B, b),
        _cb(L.CB_PARTIAL_OWN, bf16, L.BF16_TILE_BYTES, (L.OWNER_LAG + 1) * L.B_TILES_PER_CORE, b),
        _cb(L.CB_H, bf16, L.BF16_TILE_BYTES, 4 * L.B_TILES_PER_CORE, b),
        _cb(L.CB_Y, bf16, L.BF16_TILE_BYTES, 4 * L.B_TILES_PER_CORE, b),
        _cb(L.CB_SEND, bf16, L.BF16_TILE_BYTES, (L.OWNER_LAG + 1) * seg, b),
        *[_cb(L.CB_RS + i, bf16, L.BF16_TILE_BYTES, L.RING_DEPTH * seg, b) for i in range(L.CHIPS - 1)],
        _cb(L.CB_AG, bf16, L.BF16_TILE_BYTES, (L.CHIPS - 1) * L.RING_DEPTH * seg, b),
        _cb(L.CB_MESSAGE, ttnn.uint32, 64, 1, b),
        _cb(L.CB_KOWN, bf16, L.BF16_TILE_BYTES, 2 * seg, b),
        _cb(L.CB_KOUT, bf16, L.BF16_TILE_BYTES, 2 * (L.K_GROUPS - 1) * seg, b),
        *[_cb(L.CB_KRED + i, bf16, L.BF16_TILE_BYTES, L.KRED_DEPTH * seg, b) for i in range(L.K_GROUPS - 1)],
        _cb(L.CB_FLAGS, ttnn.uint32, 16 * L.FLAG_BYTES, 1, b),
        _cb(L.CB_RELAY_SLOTS, ttnn.uint32, L.RELAY_SLOT_BYTES, 2 * L.RELAY_SOURCES, relay),
    ]


def _packed(entries):
    names = {}
    flat = []
    for name, value in entries:
        names[f"rt_{name}"] = len(flat)
        flat.extend(value if isinstance(value, list) else [value])
    return names, flat


def _semaphores(chips):
    roles = [
        (L.NORM_GRID.core_range_set(), L.N_DEPTH),
        (L.a_active_cores(), L.N_DEPTH + 2),
        (L.B_GRID.core_range_set(), 16),
    ]
    return [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0)
        for cores, count in roles
        for i in range(count)
    ]


def create_program_descriptor(h, w_norm, w_gate_up, w_down, y, chip, chips, node_ids, startup):
    device = h.device()
    rows = h.shape[0] // L.TILE
    assert rows % L.NORM_CORES == 0 and rows % chips == 0
    assert chips in (1, L.CHIPS)
    owner_lag = L.OWNER_LAG if chips > 1 else 0

    def physical(x, y_):
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y_))
        return core.x, core.y

    def rect(name, block):
        (x0, y0), (x1, y1) = physical(block.x_start, block.y_start), physical(block.x_end, block.y_end)
        return {
            f"{name}_x_start": x0,
            f"{name}_y_start": y0,
            f"{name}_x_end": x1,
            f"{name}_y_end": y1,
            f"{name}_cores": block.cores,
        }

    tensors = dict(zip(TENSORS, [h, w_norm, w_gate_up, w_down, y]))
    accessor_args = []
    accessor_offsets = {}
    for name, tensor in tensors.items():
        accessor_offsets[f"ta_{name}"] = len(accessor_args)
        accessor_args += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()

    named = {
        "rows": rows,
        "chips": chips,
        "owner_lag": owner_lag,
        "reduce_lag": L.REDUCE_LAG,
        "ring_depth": L.RING_DEPTH,
        "n_depth": L.N_DEPTH,
        "h_tiles": L.HIDDEN_TILES,
        "i_tiles": L.INTERMEDIATE_TILES,
        "gate_up_tiles": L.GATE_UP_TILES,
        "tile_bytes_bf16": L.BF16_TILE_BYTES,
        "tile_bytes_bfp4": L.BFP4_TILE_BYTES,
        "tile_bytes_act": L.ACT_TILE_BYTES,
        "norm_cores": L.NORM_CORES,
        "norm_chunk_tiles": L.NORM_CHUNK_TILES,
        "norm_dest_tiles": L.NORM_DEST_TILES,
        "mcast_chunk_tiles": L.MCAST_CHUNK_TILES,
        "norm_inv_width_bits": struct.unpack("I", struct.pack("f", 1.0 / (L.HIDDEN_TILES * L.TILE)))[0],
        "norm_eps_bits": struct.unpack("I", struct.pack("f", EPS))[0],
        "a_active": L.A_ACTIVE,
        "a_tiles_per_core": L.A_TILES_PER_CORE,
        "a_columns_per_core": 2 * L.A_TILES_PER_CORE,
        "b_cores": L.B_CORES,
        "b_width": L.B_COLUMNS,
        "b_tiles_per_core": L.B_TILES_PER_CORE,
        "k_groups": L.K_GROUPS,
        "k_tiles_per_b": L.K_TILES_PER_B,
        "n_tiles_per_b": L.N_TILES_PER_B,
        "a_per_k_group": L.A_PER_K_GROUP,
        "kred_depth": L.KRED_DEPTH,
        "flag_bytes": L.FLAG_BYTES,
        "b_rows": L.B_ROWS,
        "a_rows": L.A_ROWS,
        "relay_slot_bytes": L.RELAY_SLOT_BYTES,
        "relay_sources": L.RELAY_SOURCES,
        "sem_n_valid": L.SEM_N_VALID,
        "sem_n_free": L.SEM_N_FREE,
        "sem_a_valid": L.SEM_A_VALID,
        "sem_a_free": L.SEM_A_FREE,
        "sem_rs_ready": L.SEM_RS_READY,
        "sem_ag_ready": L.SEM_AG_READY,
        "sem_credit_rs": L.SEM_CREDIT_RS,
        "sem_credit_ag": L.SEM_CREDIT_AG,
        "sem_relay_free": L.SEM_RELAY_FREE,
        **accessor_offsets,
        **rect("a_grid", L.A_GRID),
        **rect("b_grid", L.B_GRID),
        **rect("norm_grid", L.NORM_GRID),
        **rect("relay_grid", L.RELAY_GRID),
    }
    cb_names = {
        "cb_norm_weight": L.CB_NORM_WEIGHT,
        "cb_norm_in": L.CB_NORM_IN,
        "cb_norm_out": L.CB_NORM_OUT,
        "cb_norm_scaler": L.CB_NORM_SCALER,
        "cb_norm_startup": L.CB_NORM_STARTUP,
        "cb_norm_squares": L.CB_NORM_SQUARES,
        "cb_norm_sum": L.CB_NORM_SUM,
        "cb_norm_rstd": L.CB_NORM_RSTD,
        "cb_norm_scaled": L.CB_NORM_SCALED,
        "cb_n": L.CB_N,
        "cb_wg": L.CB_WG,
        "cb_a_out": L.CB_A_OUT,
        "cb_a": L.CB_A,
        "cb_wd": L.CB_WD,
        "cb_partial_own": L.CB_PARTIAL_OWN,
        "cb_h": L.CB_H,
        "cb_y": L.CB_Y,
        "cb_send": L.CB_SEND,
        "cb_rs": L.CB_RS,
        "cb_ag": L.CB_AG,
        "cb_relay_slots": L.CB_RELAY_SLOTS,
        "cb_message": L.CB_MESSAGE,
        "cb_kown": L.CB_KOWN,
        "cb_kout": L.CB_KOUT,
        "cb_kred": L.CB_KRED,
        "cb_flags": L.CB_FLAGS,
    }
    addresses = {f"{name}_addr": tensor.buffer_address() for name, tensor in tensors.items()}

    def kernel(source, cores, config, runtime_args, extra_named, names, **options):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cores,
            compile_time_args=accessor_args,
            named_compile_time_args=list({**named, **cb_names, **extra_named, **names}.items()),
            runtime_args=runtime_args,
            config=config,
            **options,
        )

    def role_kernels(cores, place, count, sources, extra_args=lambda index: []):
        runtime_args = ttnn.RuntimeArgs()
        names = None
        for index in range(count):
            x, y_ = place(index)
            names, flat = _packed([("index", index), ("chip", chip), *addresses.items(), *extra_args(index)])
            runtime_args[x][y_] = flat
        return [kernel(source, cores, config, runtime_args, {}, names) for source, config in sources]

    def data_movement(processor):
        noc = ttnn.NOC.RISCV_0_default if processor == ttnn.DataMovementProcessor.RISCV_0 else ttnn.NOC.RISCV_1_default
        return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)

    def compute(full_sync):
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=full_sync,
        )

    def b_arguments(index):
        x, y_ = physical(*L.B_GRID.logical(index))
        return [("self_x", x), ("self_y", y_)]

    risc0, risc1 = ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1
    kernels = [
        *role_kernels(
            L.NORM_GRID.core_range_set(),
            L.NORM_GRID.logical,
            L.NORM_CORES,
            [
                ("norm_in.cpp", data_movement(risc1)),
                ("norm_out.cpp", data_movement(risc0)),
                ("norm_compute.cpp", compute(False)),
            ],
        ),
        *role_kernels(
            L.a_active_cores(),
            L.a_core,
            L.A_ACTIVE,
            [
                ("a_dm_in.cpp", data_movement(risc1)),
                ("a_dm_out.cpp", data_movement(risc0)),
                ("a_compute.cpp", compute(False)),
            ],
        ),
        *role_kernels(
            L.B_GRID.core_range_set(),
            L.B_GRID.logical,
            L.B_CORES,
            [
                ("b_dm_in.cpp", data_movement(risc1)),
                ("b_dm_out.cpp", data_movement(risc0)),
                ("b_compute.cpp", compute(True)),
            ],
            b_arguments,
        ),
    ]
    relay_kernels = []
    if chips > 1:
        fabric_defines = ttnn.get_fabric_kernel_defines("Linear")
        for relay in range(L.RELAY_COUNT):
            direction, link = divmod(relay, 2)
            x, y_ = L.RELAY_GRID.logical(relay)
            runtime_args = ttnn.RuntimeArgs()
            self_x, self_y = physical(x, y_)
            names, flat = _packed(
                [
                    ("self_x", self_x),
                    ("self_y", self_y),
                    ("arrival_address", startup[0]),
                    ("launch_address", startup[1]),
                ]
            )
            runtime_args[x][y_] = flat
            relay_kernels.append(
                kernel(
                    "relay.cpp",
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y_), ttnn.CoreCoord(x, y_))]),
                    data_movement(risc0),
                    runtime_args,
                    {"relay_forward": int(direction == 0), "relay_direction": direction, "relay_link": link},
                    {**names, "rt_fabric": len(flat)},
                    defines=fabric_defines,
                )
            )
    program = ttnn.ProgramDescriptor(
        kernels=kernels + relay_kernels, semaphores=_semaphores(chips), cbs=_role_cbs(chips)
    )
    for relay, relay_kernel in enumerate(relay_kernels):
        direction, link = divmod(relay, 2)
        x, y_ = L.RELAY_GRID.logical(relay)
        step = 1 if direction == 0 else -1
        connection = ttnn.setup_fabric_connection(
            node_ids[chip], node_ids[(chip + step) % chips], link, program, ttnn.CoreCoord(x, y_)
        )
        program.kernels[len(kernels) + relay].runtime_args[x][y_].extend(
            [1, *connection, 0] if direction == 0 else [0, 1, *connection]
        )
    return program

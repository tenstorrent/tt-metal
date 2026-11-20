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


def _role_cbs():
    bf16, bfp4, fp32, act = ttnn.bfloat16, ttnn.bfloat4_b, ttnn.float32, ttnn.bfloat8_b
    t = L.HIDDEN_TILES
    seg = L.B_TILES_PER_CORE
    norm, a, b = L.core_range_set(L.NORM_CORE_LIST), L.core_range_set(L.A_CORE_LIST), L.B_GRID.core_range_set()
    return [
        _cb(L.CB_A, act, L.ACT_TILE_BYTES, L.A_DEPTH * L.INTERMEDIATE_TILES, b),
        _cb(L.CB_NORM_OUT, act, L.ACT_TILE_BYTES, L.NORM_CORES * L.N_ROW_TILES, norm),
        _cb(L.CB_N, act, L.ACT_TILE_BYTES, L.N_DEPTH * L.N_ROW_TILES, a),
        _cb(L.CB_GATHER, act, L.ACT_TILE_BYTES, L.A_DEPTH * L.GATHER_TILES, a),
        *[_cb(L.CB_LANDING + i, bf16, L.BF16_TILE_BYTES, L.LANDING_DEPTH * seg, b) for i in range(L.CHIPS - 1)],
        _cb(L.CB_RELAY_SLOTS, ttnn.uint32, L.RELAY_SLOT_BYTES, 2 * L.RELAY_SOURCE_SLOTS, b),
        _cb(L.CB_NORM_SCALER, bf16, L.BF16_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_STARTUP, bfp4, L.BFP4_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_ONES, bf16, L.BF16_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_SUM, fp32, 4096, L.NORM_DEST_TILES, norm),
        _cb(L.CB_NORM_RSTD, bf16, L.BF16_TILE_BYTES, 1, norm),
        _cb(L.CB_NORM_WEIGHT, bf16, L.BF16_TILE_BYTES, t, norm),
        _cb(L.CB_NORM_IN, bf16, L.BF16_TILE_BYTES, 2 * t, norm),
        _cb(L.CB_WG, bfp4, L.BFP4_TILE_BYTES, t * 2 * L.A_TILES_PER_CORE, a),
        _cb(L.CB_A_OUT, act, L.ACT_TILE_BYTES, L.A_OUT_DEPTH * L.A_TILES_PER_CORE, a),
        _cb(L.CB_WD, bfp4, L.BFP4_TILE_BYTES, L.INTERMEDIATE_TILES * seg, b),
        _cb(L.CB_PARTIAL_OWN, bf16, L.BF16_TILE_BYTES, (L.OWNER_LAG + 1) * seg, b),
        _cb(L.CB_H, bf16, L.BF16_TILE_BYTES, (L.OWNER_LAG + 1) * seg, b),
        _cb(L.CB_Y, bf16, L.BF16_TILE_BYTES, 2 * seg, b),
        _cb(L.CB_SEND, bf16, L.BF16_TILE_BYTES, 2 * seg, b),
        _cb(L.CB_MESSAGE, ttnn.uint32, 64, 1, b),
    ]


def _packed(entries):
    names = {}
    flat = []
    for name, value in entries:
        names[f"rt_{name}"] = len(flat)
        flat.extend(value if isinstance(value, list) else [value])
    return names, flat


def _semaphores():
    return [ttnn.SemaphoreDescriptor(id=i, core_ranges=L.GRID.core_range_set(), initial_value=0) for i in range(11)]


def _relay_sources():
    sources = [[] for _ in range(L.RELAY_COUNT)]
    for row in range(L.B_ROWS):
        for group, (_, _, relay) in L.PARTIAL_ROUTE.items():
            sources[relay].append((L.b_core(group, row), 0))
        sources[0 if row < L.B_ROWS // 2 else 1].append((L.b_core(L.OWNER_GROUP, row), 0))
        sources[3].append((L.b_core(L.OWNER_GROUP, row), 1))
    return sources


def create_program_descriptor(h, w_norm, w_gate_up, w_down, y, chip, chips, node_ids, startup):
    device = h.device()
    rows = h.shape[0] // L.TILE
    assert rows % L.NORM_CORES == 0
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
        "n_depth": L.N_DEPTH,
        "a_depth": L.A_DEPTH,
        "landing_depth": L.LANDING_DEPTH,
        "h_tiles": L.HIDDEN_TILES,
        "n_row_tiles": L.N_ROW_TILES,
        "i_tiles": L.INTERMEDIATE_TILES,
        "gate_up_tiles": L.GATE_UP_TILES,
        "tile_bytes_bf16": L.BF16_TILE_BYTES,
        "tile_bytes_bfp4": L.BFP4_TILE_BYTES,
        "tile_bytes_act": L.ACT_TILE_BYTES,
        "norm_cores": L.NORM_CORES,
        "norm_dest_tiles": L.NORM_DEST_TILES,
        "mcast_chunk_tiles": L.MCAST_CHUNK_TILES,
        "norm_inv_width_bits": struct.unpack("I", struct.pack("f", 1.0 / (L.HIDDEN_TILES * L.TILE)))[0],
        "norm_eps_bits": struct.unpack("I", struct.pack("f", EPS))[0],
        "a_active": L.A_ACTIVE,
        "a_collectors": L.A_COLLECTORS,
        "a_tiles_per_core": L.A_TILES_PER_CORE,
        "a_columns_per_core": 2 * L.A_TILES_PER_CORE,
        "b_cores": L.B_CORES,
        "b_tiles_per_core": L.B_TILES_PER_CORE,
        "relay_slot_bytes": L.RELAY_SLOT_BYTES,
        "relay_y": physical(0, L.RELAY_ROW)[1],
        "sem_n_valid": L.SEM_N_VALID,
        "sem_n_free": L.SEM_N_FREE,
        "sem_a_valid": L.SEM_A_VALID,
        "sem_a_free": L.SEM_A_FREE,
        "sem_gather_valid": L.SEM_GATHER_VALID,
        "sem_gather_free": L.SEM_GATHER_FREE,
        "gather_tiles": L.GATHER_TILES,
        "sem_land_ready": L.SEM_LAND_READY,
        "sem_credit_partial": L.SEM_CREDIT_PARTIAL,
        "sem_credit_y": L.SEM_CREDIT_Y,
        "sem_relay_free": L.SEM_RELAY_FREE,
        **{f"b_group_x{group}": physical(*L.b_core(group, 0))[0] for group in range(L.B_COLUMNS)},
        **accessor_offsets,
        **rect("a_collectors", L.Block(L.A_COLUMNS - 1, 0, L.A_COLUMNS - 1, L.GRID_ROWS - 1)),
        **rect("b_grid", L.B_GRID),
        **rect("a_main", L.Block(0, 1, L.A_COLUMNS - 1, L.GRID_ROWS - 1)),
        **rect("a_top", L.Block(2, 0, L.A_COLUMNS - 1, 0)),
        **rect("norm0", L.Block(*L.NORM_CORE_LIST[0], *L.NORM_CORE_LIST[0])),
        **rect("norm1", L.Block(*L.NORM_CORE_LIST[1], *L.NORM_CORE_LIST[1])),
    }
    cb_names = {
        "cb_norm_weight": L.CB_NORM_WEIGHT,
        "cb_norm_in": L.CB_NORM_IN,
        "cb_norm_scaler": L.CB_NORM_SCALER,
        "cb_norm_startup": L.CB_NORM_STARTUP,
        "cb_norm_ones": L.CB_NORM_ONES,
        "cb_norm_sum": L.CB_NORM_SUM,
        "cb_norm_rstd": L.CB_NORM_RSTD,
        "cb_norm_out": L.CB_NORM_OUT,
        "cb_n": L.CB_N,
        "cb_wg": L.CB_WG,
        "cb_a_out": L.CB_A_OUT,
        "cb_a": L.CB_A,
        "cb_gather": L.CB_GATHER,
        "cb_wd": L.CB_WD,
        "cb_partial_own": L.CB_PARTIAL_OWN,
        "cb_h": L.CB_H,
        "cb_y": L.CB_Y,
        "cb_send": L.CB_SEND,
        "cb_landing": L.CB_LANDING,
        "cb_relay_slots": L.CB_RELAY_SLOTS,
        "cb_message": L.CB_MESSAGE,
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

    def role_kernels(cores, entries, sources, extra_named=None):
        runtime_args = ttnn.RuntimeArgs()
        names = None
        for (x, y_), extra in entries:
            names, flat = _packed([("chip", chip), *addresses.items(), *extra])
            runtime_args[x][y_] = flat
        return [kernel(source, cores, config, runtime_args, extra_named or {}, names) for source, config in sources]

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

    def a_entry(index, core):
        row = L.a_row_cores(core[1])
        collector = row[-1]
        return [
            ("index", index),
            ("position", row.index(core)),
            ("row_cores", len(row)),
            ("row_first", L.A_CORE_LIST.index(row[0])),
            ("is_collector", int(core == collector)),
            ("collector_x", physical(*collector)[0]),
            ("row_x_start", physical(*row[0])[0]),
            ("row_y", physical(*core)[1]),
        ]

    risc0, risc1 = ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1
    relay_sources = _relay_sources()

    def source_index(relay, core, port):
        return relay_sources[relay].index((core, port))

    def b_entry(group, row):
        core = L.b_core(group, row)
        x, y_ = core
        first_column = (L.HIDDEN_TILES // L.CHIPS) * ((chip + group) % L.CHIPS) + L.B_TILES_PER_CORE * row
        extra = [("group", group), ("first_column", first_column), ("self_y", physical(x, y_)[1])]
        relay0 = relay1 = (0, 0)
        hops = 0
        if chips > 1 and group == L.OWNER_GROUP:
            relay0 = (0 if row < L.B_ROWS // 2 else 1, 0)
            relay1 = (3, 1)
        elif chips > 1:
            _, hops, relay = L.PARTIAL_ROUTE[group]
            relay0 = (relay, 0)
        extra += [
            ("hops", hops),
            ("relay_x0", physical(*L.RELAY_CORE_LIST[relay0[0]])[0]),
            ("relay_source0", source_index(relay0[0], core, relay0[1]) if chips > 1 else 0),
            ("relay_x1", physical(*L.RELAY_CORE_LIST[relay1[0]])[0]),
            ("relay_source1", source_index(relay1[0], core, 1) if chips > 1 and relay1[1] else 0),
        ]
        return core, extra

    def b_cores(groups):
        return [b_entry(group, row) for group in groups for row in range(L.B_ROWS)]

    owner_groups = [L.OWNER_GROUP] if chips > 1 else list(range(L.B_COLUMNS))
    other_groups = [group for group in range(L.B_COLUMNS) if group not in owner_groups]
    b_roles = [(owner_groups, 1, chips - 1)] + ([(other_groups, 0, 0)] if other_groups else [])

    kernels = [
        *role_kernels(
            L.core_range_set(L.NORM_CORE_LIST),
            [(core, [("index", i)]) for i, core in enumerate(L.NORM_CORE_LIST)],
            [
                ("norm_in.cpp", data_movement(risc1)),
                ("norm_out.cpp", data_movement(risc0)),
                ("norm_compute.cpp", compute(False)),
            ],
        ),
        *role_kernels(
            L.core_range_set(L.A_CORE_LIST),
            [(core, a_entry(i, core)) for i, core in enumerate(L.A_CORE_LIST)],
            [
                ("a_dm_in.cpp", data_movement(risc1)),
                ("a_dm_out.cpp", data_movement(risc0)),
                ("a_compute.cpp", compute(False)),
            ],
        ),
    ]
    for groups, owner, remote_sources in b_roles:
        kernels += role_kernels(
            L.core_range_set([L.b_core(group, row) for group in groups for row in range(L.B_ROWS)]),
            b_cores(groups),
            [("b_dm.cpp", data_movement(risc1)), ("b_compute.cpp", compute(True))],
            {"owner": owner, "remote_sources": remote_sources},
        )
    relay_kernels = []
    if chips > 1:
        fabric_defines = ttnn.get_fabric_kernel_defines("Linear")
        for relay, (x, y_) in enumerate(L.RELAY_CORE_LIST):
            direction, link = divmod(relay, 2)
            runtime_args = ttnn.RuntimeArgs()
            self_x, self_y = physical(x, y_)
            table = [v for core, port in relay_sources[relay] for v in (*physical(*core), port)]
            names, flat = _packed(
                [
                    ("self_x", self_x),
                    ("self_y", self_y),
                    ("arrival_address", startup[0]),
                    ("launch_address", startup[1]),
                    ("sources", table),
                ]
            )
            runtime_args[x][y_] = flat
            relay_kernels.append(
                kernel(
                    "relay.cpp",
                    L.core_range_set([(x, y_)]),
                    data_movement(risc0),
                    runtime_args,
                    {
                        "relay_forward": int(direction == 0),
                        "relay_sources": len(relay_sources[relay]),
                    },
                    {**names, "rt_fabric": len(flat)},
                    defines=fabric_defines,
                )
            )
    program = ttnn.ProgramDescriptor(kernels=kernels + relay_kernels, semaphores=_semaphores(), cbs=_role_cbs())
    for relay, (x, y_) in enumerate(L.RELAY_CORE_LIST[: len(relay_kernels)]):
        direction, link = divmod(relay, 2)
        step = 1 if direction == 0 else -1
        connection = ttnn.setup_fabric_connection(
            node_ids[chip], node_ids[(chip + step) % chips], link, program, ttnn.CoreCoord(x, y_)
        )
        program.kernels[len(kernels) + relay].runtime_args[x][y_].extend(
            [1, *connection, 0] if direction == 0 else [0, 1, *connection]
        )
    return program

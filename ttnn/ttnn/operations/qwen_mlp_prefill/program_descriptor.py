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


def _packed(entries):
    names = {}
    flat = []
    for name, value in entries:
        names[f"rt_{name}"] = len(flat)
        flat.extend(value if isinstance(value, list) else [value])
    return names, flat


def _routes(groups, chips):
    """Ports of every down core, the relay assigned to each port and the sources of every relay."""
    if chips == 1:
        return {}, {}, [[] for _ in range(L.RELAY_COUNT)]
    routes = {}
    for group in range(groups):
        for row in range(L.B_ROWS):
            for block in range(L.B_COLUMNS):
                if block == L.OWNER_BLOCK:
                    routes[group, block, row] = [(0, 0, row % 2 == 0), (1, 1, row % 2 == 1)]
                else:
                    direction = {1: 0, 2: row % 2, 3: 1}[block]
                    routes[group, block, row] = [(0, direction, 1)]
    load = [0] * L.RELAY_COUNT
    sources = [[] for _ in range(L.RELAY_COUNT)]
    assigned = {}
    for key, ports in routes.items():
        for port, direction, extra in ports:
            weight = 3 * (1 + extra) if key[1] == L.OWNER_BLOCK else 3
            relay = min((2 * direction, 2 * direction + 1), key=lambda r: load[r])
            load[relay] += weight
            sources[relay].append((L.b_core(*key), port))
            assigned[key, port] = relay
    return routes, assigned, sources


def create_program_descriptor(h, w_norm, w_gate_up, w_down, y, chip, chips, groups, node_ids, startup):
    device = h.device()
    total_rows = h.shape[0] // L.TILE
    assert total_rows % groups == 0
    rows = total_rows // groups
    assert chips in (1, L.CHIPS)
    assert groups in range(1, L.MAX_GROUPS + 1)
    owner_lag = L.OWNER_LAG if chips > 1 else 0
    all_groups = range(groups)

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

    def partner(core, group):
        return physical(core[0], core[1] + (L.GROUP_ROWS if group == 0 else -L.GROUP_ROWS)) if groups > 1 else (0, 0)

    tensors = dict(zip(TENSORS, [h, w_norm, w_gate_up, w_down, y]))
    accessor_args = []
    accessor_offsets = {}
    for name, tensor in tensors.items():
        accessor_offsets[f"ta_{name}"] = len(accessor_args)
        accessor_args += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()

    named = {
        "rows": rows,
        "groups": groups,
        "chips": chips,
        "owner_lag": owner_lag,
        "n_depth": L.N_DEPTH,
        "a_depth": L.A_DEPTH,
        "gather_depth": L.GATHER_DEPTH,
        "landing_depth": L.LANDING_DEPTH,
        "h_tiles": L.HIDDEN_TILES,
        "n_row_tiles": L.N_ROW_TILES,
        "i_tiles": L.INTERMEDIATE_TILES,
        "gate_up_tiles": L.GATE_UP_TILES,
        "tile_bytes_bf16": L.BF16_TILE_BYTES,
        "tile_bytes_bfp4": L.BFP4_TILE_BYTES,
        "tile_bytes_act": L.ACT_TILE_BYTES,
        "norm_chunk_tiles": L.NORM_CHUNK_TILES,
        "norm_in_chunks": L.NORM_IN_CHUNKS,
        "norm_reads_in_flight": L.NORM_READS_IN_FLIGHT,
        "norm_dest_tiles": L.NORM_DEST_TILES,
        "norm_inv_width_bits": struct.unpack("I", struct.pack("f", 1.0 / (L.HIDDEN_TILES * L.TILE)))[0],
        "norm_eps_bits": struct.unpack("I", struct.pack("f", EPS))[0],
        "a_active": L.A_ACTIVE,
        "a_tiles_per_core": L.A_TILES_PER_CORE,
        "a_columns_per_core": 2 * L.A_TILES_PER_CORE,
        "b_cores": L.B_CORES,
        "b_tiles_per_core": L.B_TILES_PER_CORE,
        "packet_tiles": L.PACKET_TILES,
        "weight_chunk_rows": L.WEIGHT_CHUNK_ROWS,
        "weight_reads_in_flight": L.WEIGHT_READS_IN_FLIGHT,
        "relay_slots": L.RELAY_SLOTS,
        "relay_slot_bytes": L.RELAY_SLOT_BYTES,
        "sem_n_valid": L.SEM_N_VALID,
        "sem_n_free": L.SEM_N_FREE,
        "sem_a_valid": L.SEM_A_VALID,
        "sem_a_free": L.SEM_A_FREE,
        "sem_gather_valid": L.SEM_GATHER_VALID,
        "sem_gather_free": L.SEM_GATHER_FREE,
        "sem_a_weights": L.SEM_A_WEIGHTS,
        "sem_b_weights": L.SEM_B_WEIGHTS,
        "gather_tiles": L.GATHER_TILES,
        "sem_land_ready": L.SEM_LAND_READY,
        "sem_credit_partial": L.SEM_CREDIT_PARTIAL,
        "sem_credit_y": L.SEM_CREDIT_Y,
        "sem_relay_free": L.SEM_RELAY_FREE,
        **{f"b_block_x{block}": physical(*L.b_core(0, block, 0))[0] for block in range(L.B_COLUMNS)},
        **accessor_offsets,
    }
    cb_names = {
        "cb_norm_weight": L.CB_NORM_WEIGHT,
        "cb_norm_in": L.CB_NORM_IN,
        "cb_norm_scaler": L.CB_NORM_SCALER,
        "cb_norm_startup": L.CB_NORM_STARTUP,
        "cb_norm_ones": L.CB_NORM_ONES,
        "cb_norm_sum": L.CB_NORM_SUM,
        "cb_norm_rstd": L.CB_NORM_RSTD,
        "cb_norm_rstd_out": L.CB_NORM_RSTD_OUT,
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

    def role_kernels(cores, entries, sources, extra_named):
        runtime_args = ttnn.RuntimeArgs()
        names = None
        for (x, y_), extra in entries:
            names, flat = _packed([("chip", chip), *addresses.items(), *extra])
            runtime_args[x][y_] = flat
        return [kernel(source, cores, config, runtime_args, extra_named, names) for source, config in sources]

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

    risc0, risc1 = ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1
    routes, assigned, relay_sources = _routes(groups, chips)

    sources_of = [set(sources) for sources in relay_sources]

    def source_index(relay, core, port):
        return relay_sources[relay].index((core, port))

    def a_entry(group, index, core):
        row = L.a_row_cores(group, core[1] - L.first_row(group))
        collector = row[-1]
        return core, [
            ("index", index),
            ("position", row.index(core)),
            ("row_cores", len(row)),
            ("row_first", L.a_cores(group).index(row[0])),
            ("is_collector", int(core == collector)),
            ("collector_x", physical(*collector)[0]),
            ("row_x_start", physical(*row[0])[0]),
            ("row_y", physical(*core)[1]),
            ("partner_x", partner(core, group)[0]),
            ("partner_y", partner(core, group)[1]),
        ]

    def b_entry(group, block, row):
        core = L.b_core(group, block, row)
        first_column = (L.HIDDEN_TILES // L.CHIPS) * ((chip + block) % L.CHIPS) + L.B_TILES_PER_CORE * row
        extra = [
            ("block", block),
            ("first_column", first_column),
            ("self_y", physical(*core)[1]),
            ("partner_x", partner(core, group)[0]),
            ("partner_y", partner(core, group)[1]),
        ]
        relay = [0, 0]
        port_offset2 = 0
        hops = 0
        if chips > 1:
            key = (group, block, row)
            for port, _, offset2 in routes[key]:
                relay[port] = assigned[key, port]
            if block == L.OWNER_BLOCK:
                port_offset2 = 0 if row % 2 == 0 else 1
            else:
                hops = 1 if block != 2 else 2
        extra += [
            ("hops", hops),
            ("port_offset2", port_offset2),
            *[
                item
                for port in range(2)
                for item in (
                    (f"relay_x{port}", physical(*L.RELAY_CORE_LIST[relay[port]])[0]),
                    (f"relay_y{port}", physical(*L.RELAY_CORE_LIST[relay[port]])[1]),
                    (
                        f"relay_source{port}",
                        source_index(relay[port], core, port) if (core, port) in sources_of[relay[port]] else 0,
                    ),
                )
            ],
        ]
        return core, extra

    owner_blocks = [L.OWNER_BLOCK] if chips > 1 else list(range(L.B_COLUMNS))
    other_blocks = [block for block in range(L.B_COLUMNS) if block not in owner_blocks]
    b_roles = [(owner_blocks, 1, chips - 1)] + ([(other_blocks, 0, 0)] if other_blocks else [])

    def b_role_cores(group, blocks):
        return [(group, block, row) for block in blocks for row in range(L.B_ROWS)]

    def group_cores(entries):
        return [L.b_core(*entry) for entry in entries]

    norm_cores = [L.norm_core(group) for group in all_groups]
    a_core_lists = [L.a_cores(group) for group in all_groups]
    a_rects = L.core_range_set([c for group in all_groups for c in [L.norm_core(group)] + L.a_cores(group)])
    a_all = L.core_range_set([c for cores in a_core_lists for c in cores])
    b_all = L.core_range_set(
        [L.b_core(group, block, row) for group in all_groups for block in range(L.B_COLUMNS) for row in range(L.B_ROWS)]
    )
    owner_all = L.core_range_set(group_cores([e for g in all_groups for e in b_role_cores(g, owner_blocks)]))
    other_all = (
        L.core_range_set(group_cores([e for g in all_groups for e in b_role_cores(g, other_blocks)]))
        if other_blocks
        else None
    )

    bf16, bfp4, fp32, act = ttnn.bfloat16, ttnn.bfloat4_b, ttnn.float32, ttnn.bfloat8_b
    t, seg = L.HIDDEN_TILES, L.B_TILES_PER_CORE
    norm_set = L.core_range_set(norm_cores)
    max_relay_sources = max(len(sources) for sources in relay_sources)
    cbs = [
        _cb(L.CB_N, act, L.ACT_TILE_BYTES, L.N_DEPTH * L.N_ROW_TILES, a_rects),
        _cb(L.CB_A, act, L.ACT_TILE_BYTES, L.A_DEPTH * L.INTERMEDIATE_TILES, b_all),
        _cb(L.CB_NORM_WEIGHT, bf16, L.BF16_TILE_BYTES, t, norm_set),
        _cb(L.CB_NORM_IN, bf16, L.BF16_TILE_BYTES, L.NORM_IN_CHUNKS * L.NORM_CHUNK_TILES, norm_set),
        _cb(L.CB_NORM_OUT, act, L.ACT_TILE_BYTES, L.NORM_OUT_CHUNKS * L.NORM_CHUNK_TILES, norm_set),
        _cb(L.CB_NORM_SCALER, bf16, L.BF16_TILE_BYTES, 1, norm_set),
        _cb(L.CB_NORM_STARTUP, bfp4, L.BFP4_TILE_BYTES, 1, norm_set),
        _cb(L.CB_NORM_ONES, bf16, L.BF16_TILE_BYTES, 1, norm_set),
        _cb(L.CB_NORM_SUM, fp32, 4096, L.NORM_DEST_TILES, norm_set),
        _cb(L.CB_NORM_RSTD, bf16, L.BF16_TILE_BYTES, 1, norm_set),
        _cb(L.CB_NORM_RSTD_OUT, act, L.ACT_TILE_BYTES, 1, norm_set),
        _cb(L.CB_WG, bfp4, L.BFP4_TILE_BYTES, t * 2 * L.A_TILES_PER_CORE, a_all),
        _cb(L.CB_A_OUT, act, L.ACT_TILE_BYTES, L.A_OUT_DEPTH * L.A_TILES_PER_CORE, a_all),
        _cb(L.CB_GATHER, act, L.ACT_TILE_BYTES, L.GATHER_DEPTH * L.GATHER_TILES, a_all),
        *[_cb(L.CB_LANDING + i, bf16, L.BF16_TILE_BYTES, L.LANDING_DEPTH * seg, owner_all) for i in range(L.CHIPS - 1)],
        *(
            [
                _cb(L.CB_LANDING, bf16, L.BF16_TILE_BYTES, L.LANDING_DEPTH * seg, other_all),
                _cb(L.CB_RELAY_SLOTS, ttnn.uint32, L.RELAY_SLOT_BYTES, L.RELAY_SLOTS * max_relay_sources, other_all),
            ]
            if other_all is not None
            else []
        ),
        _cb(L.CB_WD, bfp4, L.BFP4_TILE_BYTES, L.INTERMEDIATE_TILES * seg, b_all),
        _cb(L.CB_PARTIAL_OWN, bf16, L.BF16_TILE_BYTES, (owner_lag + 1) * seg, owner_all),
        _cb(L.CB_H, bf16, L.BF16_TILE_BYTES, 2 * seg, owner_all),
        _cb(L.CB_Y, bf16, L.BF16_TILE_BYTES, 2 * seg, owner_all),
        *([_cb(L.CB_SEND, bf16, L.BF16_TILE_BYTES, 2 * seg, other_all)] if other_all is not None else []),
        _cb(L.CB_MESSAGE, ttnn.uint32, 64, 1, b_all),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0)
        for cores, count in [
            (norm_set, L.N_DEPTH),
            (a_all, L.SEM_A_WEIGHTS + 1),
            (b_all, L.SEM_B_WEIGHTS + 1),
        ]
        for i in range(count)
    ]

    kernels = []
    for group in all_groups:
        group_named = {
            "group": group,
            **rect("a_rect", L.a_rect(group)),
            **rect("b_rect", L.b_rect(group)),
            **rect("norm", L.Block(*L.norm_core(group), *L.norm_core(group))),
            **rect("a_collectors", L.a_collectors(group)),
        }
        kernels += role_kernels(
            L.core_range_set([L.norm_core(group)]),
            [
                (
                    L.norm_core(group),
                    [("self_x", physical(*L.norm_core(group))[0]), ("self_y", physical(*L.norm_core(group))[1])],
                )
            ],
            [
                ("norm_in.cpp", data_movement(risc1)),
                ("norm_out.cpp", data_movement(risc0)),
                ("norm_compute.cpp", compute(False)),
            ],
            group_named,
        )
        kernels += role_kernels(
            L.core_range_set(L.a_cores(group)),
            [a_entry(group, i, core) for i, core in enumerate(L.a_cores(group))],
            [
                ("a_dm_in.cpp", data_movement(risc1)),
                ("a_dm_out.cpp", data_movement(risc0)),
                ("a_compute.cpp", compute(False)),
            ],
            group_named,
        )
        for blocks, owner, remote_sources in b_roles:
            entries = b_role_cores(group, blocks)
            kernels += role_kernels(
                L.core_range_set(group_cores(entries)),
                [b_entry(*entry) for entry in entries],
                [("b_dm.cpp", data_movement(risc1)), ("b_compute.cpp", compute(True))],
                {
                    **group_named,
                    "owner": owner,
                    "remote_sources": remote_sources,
                },
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
                        "cb_relay_slots": L.CB_RELAY_SLOTS,
                    },
                    {**names, "rt_fabric": len(flat)},
                    defines=fabric_defines,
                )
            )
    program = ttnn.ProgramDescriptor(kernels=kernels + relay_kernels, semaphores=semaphores, cbs=cbs)
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

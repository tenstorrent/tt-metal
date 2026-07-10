#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model one synchronized ring round with tt-npe's congestion model.

Source tt-npe's ENV_SETUP before running this script.  The workload contains
one simultaneous neighbor write per active Tensix, matching one payload hop in
the device microbenchmark.  The reported full-round estimate multiplies that
hop by ``group_size - 1``; semaphores and compute are intentionally absent.
Architecture, grid shape, and worker coordinates are queried from the device.
"""

import argparse
import math

import tt_npe_pybind as npe
import tt_umd
import ttnn

NPE_DEVICE_INDEX = 0


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _device_topology(pci_device_id):
    device = tt_umd.TTDevice.create(pci_device_id)
    device.init_tt_device()
    soc = tt_umd.SocDescriptor(device)
    logical_cores = soc.get_cores(tt_umd.CoreType.TENSIX, tt_umd.CoordSystem.LOGICAL)
    if not logical_cores:
        raise RuntimeError(f"device {pci_device_id} has no active Tensix cores")

    worker_coords = {}
    for logical in logical_cores:
        physical = soc.translate_coord_to(logical, tt_umd.CoordSystem.NOC0)
        worker_coords[(logical.x, logical.y)] = (physical.x, physical.y)
    grid_shape = (
        max(core.y for core in logical_cores) + 1,
        max(core.x for core in logical_cores) + 1,
    )
    return device.get_arch().name.lower(), grid_shape, worker_coords


def _groups(group_shape, num_groups, grid_shape):
    rows, cols = group_shape
    grid_rows, grid_cols = grid_shape
    groups_across = grid_cols // cols
    capacity = groups_across * (grid_rows // rows)
    if groups_across == 0 or num_groups > capacity:
        raise ValueError(
            f"cannot place {num_groups} groups of {rows}x{cols} on the {grid_rows}x{grid_cols} worker grid"
        )
    result = []
    for group_index in range(num_groups):
        gx = (group_index % groups_across) * cols
        gy = (group_index // groups_across) * rows
        ring = []
        for y in range(rows):
            xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
            ring.extend((gx + x, gy + y) for x in xs)
        result.append(ring)
    return result


def _transfer_shape(payload_bytes, requested_packet_bytes):
    packet_bytes = payload_bytes if requested_packet_bytes is None else math.gcd(payload_bytes, requested_packet_bytes)
    return packet_bytes, payload_bytes // packet_bytes


def estimate(
    group_shape,
    num_groups,
    payload_bytes,
    noc,
    *,
    device_name,
    grid_shape,
    worker_coords,
    congestion_model=None,
    cycles_per_timestep=None,
    packet_bytes=None,
):
    phase = npe.Phase()
    packet_bytes, num_packets = _transfer_shape(payload_bytes, packet_bytes)
    for ring in _groups(group_shape, num_groups, grid_shape):
        for index, (src_x, src_y) in enumerate(ring):
            dst_x, dst_y = ring[(index + 1) % len(ring)]
            src_physical_x, src_physical_y = worker_coords[(src_x, src_y)]
            dst_physical_x, dst_physical_y = worker_coords[(dst_x, dst_y)]
            src = npe.Coord(NPE_DEVICE_INDEX, src_physical_y, src_physical_x)
            dst = npe.Coord(NPE_DEVICE_INDEX, dst_physical_y, dst_physical_x)
            noc_type = npe.NocType.NOC_0 if noc == 0 else npe.NocType.NOC_1
            phase.addTransfer(npe.Transfer(packet_bytes, num_packets, src, dst, 0.0, 0, noc_type))

    workload = npe.Workload()
    workload.addPhase(phase)
    # Required by the current API; only model estimates are reported below.
    workload.setGoldenResultCycles({NPE_DEVICE_INDEX: (0, 1)})
    config = npe.Config()
    config.device_name = device_name
    if congestion_model is not None:
        config.congestion_model_name = congestion_model
    if cycles_per_timestep is not None:
        config.cycles_per_timestep = cycles_per_timestep
    result = npe.InitAPI(config).runNPE(workload)
    if type(result) is not npe.Stats:
        raise RuntimeError(f"tt-npe failed: {result}")
    return result.per_device_stats[NPE_DEVICE_INDEX]


def _default_cases(grid_shape):
    grid_rows, grid_cols = grid_shape
    half_row_cols = max(2, grid_cols // 2)
    candidates = (
        ("half_rows", (1, half_row_cols), (grid_cols // half_row_cols) * grid_rows),
        ("whole_rows", (1, grid_cols), grid_rows),
        ("whole_columns", (grid_rows, 1), grid_cols),
        ("two_rows", (min(2, grid_rows), grid_cols), grid_rows // min(2, grid_rows)),
    )
    return tuple(case for case in candidates if case[1][0] * case[1][1] >= 2 and case[2] >= 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tiles", nargs="+", type=_positive, default=[1, 6, 24])
    parser.add_argument("--group-shape", help="optional ROWS,COLS instead of the default matrix")
    parser.add_argument("--num-groups", type=_positive, default=1)
    parser.add_argument("--noc", choices=("0", "1", "both"), default="both")
    parser.add_argument(
        "--pci-device-id",
        type=int,
        help="PCI device used to discover arch, grid, and worker mapping; default: first enumerated device",
    )
    parser.add_argument("--congestion-model", help="override tt-npe's default congestion model")
    parser.add_argument("--cycles-per-timestep", type=_positive, help="override tt-npe's default model timestep")
    parser.add_argument(
        "--packet-bytes",
        type=_positive,
        help="optional packetization override; default models each kernel transfer as one packet",
    )
    args = parser.parse_args()

    pci_device_id = args.pci_device_id
    if pci_device_id is None:
        pci_device_id = next(iter(tt_umd.PCIDevice.enumerate_devices()), None)
        if pci_device_id is None:
            parser.error("no PCI devices are available for topology discovery")

    device_name, grid_shape, worker_coords = _device_topology(pci_device_id)

    if args.group_shape:
        try:
            rows, cols = (int(value) for value in args.group_shape.split(","))
        except ValueError:
            parser.error("--group-shape must be ROWS,COLS")
        if rows < 1 or cols < 1:
            parser.error("--group-shape dimensions must be positive")
        cases = (("custom", (rows, cols), args.num_groups),)
    else:
        cases = _default_cases(grid_shape)

    print(f"device={device_name}  worker-grid={grid_shape[0]}x{grid_shape[1]}")
    print("| NoC | Placement | Group | Groups | Payload | Hop cycles | No-congestion | Congestion | Ring cycles |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    nocs = (0, 1) if args.noc == "both" else (int(args.noc),)
    for name, group_shape, num_groups in cases:
        group_size = group_shape[0] * group_shape[1]
        for num_tiles in args.num_tiles:
            payload_bytes = num_tiles * ttnn.tile_size(ttnn.bfloat16)
            for noc in nocs:
                stats = estimate(
                    group_shape,
                    num_groups,
                    payload_bytes,
                    noc,
                    device_name=device_name,
                    grid_shape=grid_shape,
                    worker_coords=worker_coords,
                    congestion_model=args.congestion_model,
                    cycles_per_timestep=args.cycles_per_timestep,
                    packet_bytes=args.packet_bytes,
                )
                print(
                    f"| {noc} | {name} | {group_shape[0]}x{group_shape[1]} | {num_groups} | {payload_bytes} B | "
                    f"{stats.estimated_cycles} | {stats.estimated_cong_free_cycles} | "
                    f"{stats.getCongestionImpact():.1f}% | {stats.estimated_cycles * (group_size - 1)} |"
                )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for div (op_design.md: Blocking Model, Work Distribution).

One flat tile axis (`leading`, `tile_row`, `tile_col` folded): page i of a, of b (same-shape regime) and of
the output are the same tile. The grid takes whole rounds of C tiles; inside a cluster, under the thread
plan (R, C, W) on STRIDED rings,

    reader hart p   reads    tiles p, p+R, ...   (a, and b when the divisor is a tensor)
    NEO t           computes tiles t, t+C, ...   `block_tiles` of them per chain call
    writer hart w   writes   tiles w, w+W, ...

Regimes: `eltwise_same_shape` (b a tensor, ring `in_b`) and `eltwise_number` (b a Python number, sent as its
float32 bits in the runtime argument `divisor_bits`, no ring and no tensor). The regime is a define
(`DIVISOR_IS_NUMBER`) so a binary holds one chain and names only the rings it is bound to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Knobs (each defined once; every dependent quantity derives from them) -------------------------------
# Thread plan (R readers, C NEOs, W writers): compute-bound (SFPU reciprocal + Newton steps per element),
# two inputs per output tile -> (4, 4, 2) (target page, *Choosing the plan*).
THREAD_PLAN = (4, 4, 2)
# Tiles one NEO takes from each ring per chain call (`flat_tile` block extent).
BLOCK_TILES = 4
# Blocks per NEO in flight on each ring (2 = the reader fills block k+1 while the NEO computes block k).
BUFFER_DEPTH = 2

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
NEOS = 4
RING_MAX_BYTES = 1_048_560  # a DFB ring on the compute side: 1 MiB - 16 B

ROUNDING_MODE_ID = {None: 0, "floor": 1, "trunc": 2}


def validate_thread_plan(threads):
    r, c, w = threads
    if not (1 <= c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"div: thread plan (R, C, W) = {threads}: need C <= {NEOS}, R + W <= {FREE_DM_HARTS}, R | C, W | C")


def split_rounds(num_tiles, grid_size, compute_threads):
    """(core set, {core: tiles}, {core: first tile}): whole rounds of C tiles over the grid, row-major.

    Only the last engaged cluster can hold a partial round."""
    num_rounds = -(-num_tiles // compute_threads)
    _, all_cores, group_1, group_2, rounds_1, rounds_2 = ttnn.split_work_to_cores(grid_size, num_rounds, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, rounds in ((group_1, rounds_1), (group_2, rounds_2)):
        if rounds == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            n = min(rounds * compute_threads, num_tiles - start)
            counts[core], starts[core] = n, start
            start += n
    return all_cores, counts, starts


def float32_bits(value: float) -> int:
    """The float32 bit pattern of a Python number; saturates to +-inf past FLT_MAX instead of raising."""
    with np.errstate(over="ignore"):
        return int(np.array(value, dtype=np.float32).view(np.uint32))


def create_program_artifacts(
    a,
    b,
    out,
    *,
    divisor=None,
    rounding_mode=None,
    compute_kernel_config,
    threads=THREAD_PLAN,
    block_tiles=BLOCK_TILES,
    buffer_depth=BUFFER_DEPTH,
    tag_threads=False,
):
    """`b` is the divisor tensor, or None with `divisor` the Python number.

    `tag_threads` is a test probe: odd NEOs multiply instead of divide, so a test sees which NEO computed which
    tile (a correct result alone does not show that every NEO ran)."""
    validate_thread_plan(threads)
    R, C, W = threads
    number = b is None
    if number == (divisor is None):
        raise ValueError("div: pass exactly one of the divisor tensor and the divisor number")

    num_tiles = out.buffer_num_pages()
    core_set, counts, starts = split_rounds(num_tiles, a.device().compute_with_storage_grid_size(), C)

    # A STRIDED ring is sliced into one sub-ring per NEO, so its capacity is counted per NEO.
    entries = buffer_depth * block_tiles * C
    rings = [("in_a", a)] + ([] if number else [("in_b", b)]) + [("out", out)]
    dfbs = []
    for name, tensor in rings:
        page_bytes = tensor.buffer_page_size()
        if entries * page_bytes > RING_MAX_BYTES:
            raise ValueError(f"div: ring {name} of {entries} x {page_bytes} B exceeds {RING_MAX_BYTES} B")
        dfbs.append(
            E.DataflowBufferSpec(
                unique_id=name, entry_size=page_bytes, num_entries=entries, data_format_metadata=tensor.dtype
            )
        )

    tensor_names = [("a", a)] + ([] if number else [("b", b)]) + [("out", out)]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in tensor_names]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    defines = E.KernelSpec.CompilerOptions(defines={"DIVISOR_IS_NUMBER": "1" if number else "0"})
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "div_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        compiler_options=defines,
        dfb_bindings=[E.producer_of("in_a", "in_a")] + ([] if number else [E.producer_of("in_b", "in_b")]),
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="a", accessor_name="a")]
        + ([] if number else [E.TensorBinding(tensor_parameter_name="b", accessor_name="b")]),
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    precision = E.Precision.Approximate if compute_kernel_config.math_approx_mode else E.Precision.Precise
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "div_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            # Always set: ttsim aborts on a compute kernel without a fidelity (library gaps).
            fpu_math_fidelity=compute_kernel_config.math_fidelity,
            sfpu_precision_mode=precision,
            enable_32_bit_dest=bool(compute_kernel_config.fp32_dest_acc_en),
        ),
        num_threads=C,
        compiler_options=defines,
        dfb_bindings=[E.consumer_of("in_a", "in_a")]
        + ([] if number else [E.consumer_of("in_b", "in_b")])
        + [E.producer_of("out", "out")],
        compile_time_args={
            "compute_threads": C,
            "block_tiles": int(block_tiles),
            "rounding_mode": ROUNDING_MODE_ID[rounding_mode],
            "tag_threads": int(bool(tag_threads)),
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["num_tiles"] + (["divisor_bits"] if number else [])
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "div_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of("out", "out")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="div",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    dm_args = {"num_tiles": counts, "start_id": starts}
    compute_args = {"num_tiles": counts}
    if number:
        bits = float32_bits(divisor)
        compute_args["divisor_bits"] = {core: bits for core in counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values=compute_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {name: i for i, (name, _) in enumerate(tensor_names)}

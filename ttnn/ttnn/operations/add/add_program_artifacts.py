# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for add (op_design.md, Regimes).

The output's flat page axis is split in contiguous ranges over the full grid (rounds of C pages per cluster) in
every regime; the writer stores the cluster's out pages in order. What differs is where b's tiles come from:

    stream_flat       broadcast "none": page p of out = page p of a + page p of b (identical padded sequences).
                      Inside a cluster the STRIDED rings deal the range round-robin over the thread plan:
                        reader hart r reads pages r, r+R, ...; NEO t adds pages t, t+C, ...; writer w writes w, w+W, ...
    bcast_leading     broadcast "over_leading": as stream_flat, but b's page of out page p is remapped
                      (b page = nb(n) * Ht*Wt + p mod Ht*Wt; nb folds b's size-1 leading dimensions to 0).
    bcast_resident_b  broadcast "over_w" (BroadcastDim::Col), "over_h" (Row), "over_hw" (Scalar): the cluster's
                      range unfolds into runs of out pages that read one b window (`bcast_runs`, mirrored in
                      kernels/add_bcast_walk.hpp); each window is read once into b's ring and held resident
                      (caller-managed wait/pop) for every output tile that reuses it. a and out still sync in
                      uniform blocks of `block_tiles`; compute adds each run's part of a block in one helper call.
    number_scalar     broadcast "number" (R7): no b tensor and no b ring; the number's float32 bits are a runtime
                      arg filled into DEST and added on the SFPU (fp32 DEST), so it is never rounded first.

The operands are swapped when a is the operand that broadcasts (FPU broadcast is srcB-only; addition commutes, R4).

Knobs (each defined once here; every ring size, loop count and grid size derives from them):
    THREAD_PLAN         (R, C, W)
    BLOCK_TILES         tiles a NEO takes from the rings per block (one DEST section)
    RING_DEPTH_BLOCKS   blocks of capacity per NEO in each ring (and b windows of capacity in a resident ring)
    MIN_TILES_PER_CORE  smallest page count worth engaging a cluster for
    BCAST_WINDOW_TILES  most b tiles a never-wrapping Row ring holds (past it Row windows become one tile)
"""

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Knobs (op_design.md, Parameters / Blocking Model) ---------------------------------------------------
THREAD_PLAN = (1, 1, 1)  # (R, C, W): data-movement bound, one FPU op per tile (op_design.md, Work Distribution)
BLOCK_TILES = 4  # one DEST section in every DEST mode (fp32 half-sync: 4)
RING_DEPTH_BLOCKS = 2  # double buffering: read block k+1 while adding block k
MIN_TILES_PER_CORE = 1  # occupancy lamp: engage ceil(num_tiles / MIN_TILES_PER_CORE) clusters at most
BCAST_WINDOW_TILES = 256  # never-wrapping Row b ring cap: 512 KiB of bf16, half the ring cap

# --- Cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them ---
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 is the remapper
COMPUTE_THREAD_COUNTS = (1, 2, 4)  # C: one compute thread per NEO, 4 NEOs per cluster
MAX_RING_BYTES = 1_048_560  # one DFB ring, compute side

# Semantic ring names (shared with the kernels as dfb::<name>).
CB_INPUT_A = "cb_input_a_tiles"
CB_INPUT_B = "cb_input_b_tiles"
CB_OUTPUT = "cb_output_tiles"

# Regimes: the compile-time `bcast_mode` of all three kernels (kernels/add_bcast_walk.hpp, same values).
MODE_FLAT, MODE_COL, MODE_ROW, MODE_SCALAR, MODE_LEADING, MODE_NUMBER = range(6)
RESIDENT_B_MODES = (MODE_COL, MODE_ROW, MODE_SCALAR)
TILE = 32


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def needs_broadcast(shape, out_shape):
    """True if `shape` (padded with leading 1s) has a size-1 dimension where the output's is larger."""
    shape = (1,) * (len(out_shape) - len(shape)) + tuple(shape)
    return any(s == 1 and o > 1 for s, o in zip(shape, out_shape))


def plan_operands(a, b):
    """(swap, mode, geometry) of a call. `swap`: b broadcasts as srcB only, so when a is the operand that
    broadcasts the two are exchanged (addition commutes, R4). `geometry` is what the kernels' walk needs:
    the output's tile grid per image (Ht, Wt) and b's leading-dimension fold
    nb(n) = (n // n_inner) * s_outer + (n % n_inner) * s_inner (b images, a size-1 leading dim of b gives 0)."""
    if _is_number(b):
        return False, MODE_NUMBER, None
    sa, sb = tuple(a.shape), tuple(b.shape)
    out = tuple(int(d) for d in torch.broadcast_shapes(sa, sb))
    a_bc, b_bc = needs_broadcast(sa, out), needs_broadcast(sb, out)
    if a_bc and b_bc:
        raise NotImplementedError(
            f"add: both operands broadcast ({list(sa)} and {list(sb)}); one operand must have the output's shape"
        )
    swap = a_bc
    small = sa if swap else sb
    out4 = (1,) * (4 - len(out)) + out if len(out) < 4 else out
    s4 = (1,) * (len(out4) - len(small)) + tuple(small)
    h_bc = s4[-2] == 1 and out4[-2] > 1
    w_bc = s4[-1] == 1 and out4[-1] > 1
    lead_bc = any(s == 1 and o > 1 for s, o in zip(s4[:-2], out4[:-2]))
    if h_bc and w_bc:
        mode = MODE_SCALAR
    elif w_bc:
        mode = MODE_COL
    elif h_bc:
        mode = MODE_ROW
    elif lead_bc:
        mode = MODE_LEADING
    else:
        mode = MODE_FLAT
    o0, o1 = out4[0], out4[1]
    b0, b1 = s4[0], s4[1]
    geometry = {
        "Ht": -(-out4[-2] // TILE),
        "Wt": -(-out4[-1] // TILE),
        "n_inner": o1,
        "s_outer": b1 if b0 == o0 else 0,
        "s_inner": 1 if b1 == o1 else 0,
    }
    return swap, mode, geometry


def _nb(geometry, n):
    return (n // geometry["n_inner"]) * geometry["s_outer"] + (n % geometry["n_inner"]) * geometry["s_inner"]


def bcast_runs(mode, geometry, start, count, uniform=False):
    """The resident-b walk of one cluster, mirrored line for line by kernels/add_bcast_walk.hpp.

    Yields (first_out_page, length, open, close, win_first_b_page, win_tiles, lo): a stretch of consecutive out
    pages that read one b window, opened (read once, waited) at its first run and closed (popped) after its
    last; Row's b tile for out page p is p % Wt - lo, Col's and Scalar's the window's one tile.
        Scalar (over_hw): one b tile per run of images with the same nb.
        Col (over_w):     one b tile per tile-row piece.
        Row (over_h):     b's row, held across every row of a run of images with the same nb; a run inside one
                          row (or a head and a tail that do not overlap) reads only its own columns. `uniform`:
                          one run and one-tile window per out page (fallback when b's ring cannot hold every
                          window of the cluster without wrapping).
    """
    Ht, Wt = geometry["Ht"], geometry["Wt"]
    img = Ht * Wt
    end = start + count
    p = start

    def run_end(p):
        n = p // img
        nb = _nb(geometry, n)
        q = (n + 1) * img
        while q < end and _nb(geometry, q // img) == nb:
            q += img
        return nb, min(q, end)

    while p < end:
        if mode == MODE_SCALAR:
            nb, q = run_end(p)
            yield (p, q - p, True, True, nb, 1, 0)
            p = q
        elif mode == MODE_COL:
            n, r = p // img, (p // Wt) % Ht
            ln = min(Wt - p % Wt, end - p)
            yield (p, ln, True, True, _nb(geometry, n) * Ht + r, 1, 0)
            p += ln
        elif mode == MODE_ROW:
            nb, q = run_end(p)
            c0 = p % Wt
            if uniform:
                for i in range(p, q):
                    yield (i, 1, True, True, nb * Wt + i % Wt, 1, i % Wt)
                p = q
                continue
            head = Wt - c0 if c0 else 0
            if q - p <= (head if head else Wt):
                yield (p, q - p, True, True, nb * Wt + c0, q - p, c0)
                p = q
                continue
            k, tail = divmod(q - p - head, Wt)
            if k > 0 or tail >= c0:
                if head:
                    yield (p, head, True, k == 0 and tail == 0, nb * Wt, Wt, 0)
                if k:
                    yield (p + head, k * Wt, head == 0, tail == 0, nb * Wt, Wt, 0)
                if tail:
                    yield (p + head + k * Wt, tail, head == 0 and k == 0, True, nb * Wt, Wt, 0)
            else:
                yield (p, head, True, True, nb * Wt + c0, head, c0)
                yield (p + head, tail, True, True, nb * Wt, tail, 0)
            p = q
        else:
            raise ValueError(f"bcast_runs: mode {mode} has no resident-b walk")


def b_pages_read(mode, geometry, start, count, uniform=False):
    """b pages the cluster's reader fetches (structural count; the reader issues exactly these reads)."""
    if mode in (MODE_FLAT, MODE_LEADING):
        return count
    if mode == MODE_NUMBER:
        return 0
    return sum(r[5] for r in bcast_runs(mode, geometry, start, count, uniform) if r[2])


def resident_b_ring(mode, geometry, counts, starts, block, ring_depth, window_cap=BCAST_WINDOW_TILES):
    """(b ring entries, uniform) of a resident-b regime.

    A window compute waits at once must not straddle the ring's end. A Row
    ring holds every window its cluster opens, so it never wraps, while that total fits `window_cap`; past it
    the Row windows become one tile (`uniform`), like Col's and Scalar's, and one-tile windows never straddle.
    The reader fills b's ring with explicit sync and a's with implicit sync, whose credits arrive per batch (at
    most half a's ring, `ring_depth * block` entries). For compute's wait on an a block to be met, the reader must
    get past every b window up to the end of that batch while compute still holds the current window: at most one
    one-tile window per out page, so `ring_depth * block + 1` entries."""
    if mode == MODE_ROW:
        total = max(b_pages_read(mode, geometry, starts[c], counts[c]) for c in counts)
        if total <= window_cap:
            return total, False
        return ring_depth * block + 1, True
    return ring_depth * block + 1, False


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in COMPUTE_THREAD_COUNTS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C in {COMPUTE_THREAD_COUNTS}, R + W <= {FREE_DM_HARTS}, R | C, W | C"
        )


def split_pages(num_tiles, grid_size, compute_threads, min_tiles_per_core=MIN_TILES_PER_CORE):
    """(core set, {core: pages}, {core: first page}) over the clusters that get work.

    The split is in rounds of C pages (whole rounds per cluster, so each NEO's share differs by at most one
    page), converted back to pages; the last cluster's last round may be short.
    """
    rounds = -(-num_tiles // compute_threads)
    min_rounds = max(1, -(-min_tiles_per_core // compute_threads))
    max_cores = grid_size.x * grid_size.y
    num_cores = min(max_cores, -(-rounds // min_rounds))
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid_size, True)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(cores, rounds, row_wise=True)
    counts, starts, start_round = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            first = start_round * compute_threads
            counts[core] = min(per_core * compute_threads, num_tiles - first)
            starts[core] = first
            start_round += per_core
    return all_cores, counts, starts


def dest_section_tiles(config):
    """Tiles in one DEST section for this config (mirrors kernel_lib/dest_helpers.hpp get_dest_limit)."""
    full_sync = bool(config.dst_full_sync_en)
    fp32 = bool(config.fp32_dest_acc_en)
    return (8 if fp32 else 16) if full_sync else (4 if fp32 else 8)


def compute_hw_config(config):
    """ComputeConfigDescriptor -> ComputeHardwareConfig. fpu_math_fidelity is always set."""
    return E.ComputeHardwareConfig(
        fpu_math_fidelity=config.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if config.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(config.fp32_dest_acc_en),
        double_buffer_dest=not bool(config.dst_full_sync_en),
    )


def create_program_artifacts(
    a, b, out, config, *, threads=THREAD_PLAN, block_tiles=BLOCK_TILES, ring_depth=RING_DEPTH_BLOCKS,
    min_tiles_per_core=MIN_TILES_PER_CORE, window_cap=BCAST_WINDOW_TILES
):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for one `ttnn.generic_op` call.

    `b` is a tensor or a Python number. The tensor list handed to `generic_op` is `[a, b, out]`, or `[a, out]`
    for a number (`io_tensors`); the returned indices map the kernels' tensor names onto it, so a swap of the
    operands (a broadcasts) is only a swap of indices. The keyword overrides exist so tests and the perf pass
    can turn each knob without editing code.
    """
    validate_thread_plan(threads)
    swap, mode, geometry = plan_operands(a, b)
    R, C, W = threads
    if mode in RESIDENT_B_MODES and threads != (1, 1, 1):
        # The resident b window is consumed by one NEO; dealing it to C > 1 NEOs needs the ALL access pattern
        # (target page, Thread plan), which this op does not build.
        raise ValueError(f"add: broadcast regime {mode} runs at the thread plan (1, 1, 1), not {threads}")
    if ring_depth < 2:
        raise ValueError(f"ring_depth={ring_depth}: need >= 2 blocks per NEO")
    full, small = (b, a) if swap else (a, b)

    num_tiles = out.buffer_num_pages()
    core_set, counts, starts = split_pages(
        num_tiles, a.device().compute_with_storage_grid_size(), C, min_tiles_per_core
    )

    # Clamp the block once: never larger than one DEST section divided by the chain's DEST lanes per tile (the
    # number chain holds a and the filled number: 2 lanes; mechanism cap, the kernel static_asserts it), nor than
    # a NEO's share on the busiest cluster.
    lanes = 2 if mode == MODE_NUMBER else 1
    max_core_tiles = max(counts.values())
    block = max(1, min(block_tiles, dest_section_tiles(config) // lanes, -(-max_core_tiles // C)))
    entries = ring_depth * block * C  # a STRIDED ring is sliced per NEO: capacity is counted per NEO

    rings = [(CB_INPUT_A, full, entries)]
    uniform = False
    if mode != MODE_NUMBER:
        # A streamed b ring is sized like a's; a resident one by `resident_b_ring`.
        b_entries = entries
        if mode in RESIDENT_B_MODES:
            b_entries, uniform = resident_b_ring(mode, geometry, counts, starts, block, ring_depth, window_cap)
        rings.append((CB_INPUT_B, small, b_entries))
    rings.append((CB_OUTPUT, out, entries))
    footprint = 0
    for name, t, n in rings:
        ring_bytes = n * t.buffer_page_size()
        if ring_bytes > MAX_RING_BYTES:
            raise ValueError(f"add: ring {name} of {ring_bytes} B exceeds {MAX_RING_BYTES} B")
        footprint += ring_bytes
    budget = ttnn.get_max_worker_l1_unreserved_size()
    if footprint > budget:
        raise ValueError(f"add: ring footprint {footprint} B exceeds the L1 budget {budget} B")

    dfbs = [
        E.DataflowBufferSpec(unique_id=name, entry_size=t.buffer_page_size(), num_entries=n, data_format_metadata=t.dtype)
        for name, t, n in rings
    ]
    if mode == MODE_NUMBER:
        tensors, indices = (("input_a", a), ("output", out)), {"input_a": 0, "output": 1}
    else:
        tensors = (("input_a", full), ("input_b", small), ("output", out))
        indices = {"input_a": 1, "input_b": 0, "output": 2} if swap else {"input_a": 0, "input_b": 1, "output": 2}
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in tensors]

    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    walk_ct = {"bcast_mode": mode, "uniform_windows": int(uniform)}
    geometry = geometry or {"Ht": 1, "Wt": 1, "n_inner": 1, "s_outer": 0, "s_inner": 0}
    per_core = lambda value: {core: int(value) for core in counts}  # noqa: E731
    geom_args = {k: per_core(v) for k, v in geometry.items()}
    walk_rt = list(geometry)
    number_options = (
        E.KernelSpec.CompilerOptions(defines={"ADD_B_IS_NUMBER": "1"}) if mode == MODE_NUMBER else E.KernelSpec.CompilerOptions()
    )

    reader_dfbs = [E.producer_of(CB_INPUT_A, CB_INPUT_A)]
    reader_tensors = [E.TensorBinding(tensor_parameter_name="input_a", accessor_name="input_a")]
    compute_dfbs = [E.consumer_of(CB_INPUT_A, CB_INPUT_A)]
    if mode != MODE_NUMBER:
        reader_dfbs.append(E.producer_of(CB_INPUT_B, CB_INPUT_B))
        reader_tensors.append(E.TensorBinding(tensor_parameter_name="input_b", accessor_name="input_b"))
        compute_dfbs.append(E.consumer_of(CB_INPUT_B, CB_INPUT_B))
    compute_dfbs.append(E.producer_of(CB_OUTPUT, CB_OUTPUT))

    reader_hw = E.create_reader_datamovement_config()
    if mode in RESIDENT_B_MODES:
        # b's windows are pushed explicitly (add_reader.cpp says why); a stays on implicit sync.
        reader_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(
            disable_dfb_implicit_sync_for=[CB_INPUT_B]
        )
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "add_reader.cpp"),
        hw_config=reader_hw,
        num_threads=R,
        compiler_options=number_options,
        dfb_bindings=reader_dfbs,
        tensor_bindings=reader_tensors,
        compile_time_args={**plan, **walk_ct},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id", *walk_rt]),
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "add_compute.cpp"),
        hw_config=compute_hw_config(config),
        num_threads=C,
        compiler_options=number_options,
        dfb_bindings=compute_dfbs,
        compile_time_args={**plan, **walk_ct, "block_tiles": int(block)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["num_tiles", "start_id", "number_bits", *walk_rt]
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "add_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(CB_OUTPUT, CB_OUTPUT)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )
    spec = E.ProgramSpec(
        name="add",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    # R7: the number as its float32 value, bit for bit (a Python float rounded once to float32, never to bf16).
    number_bits = struct.unpack("<I", struct.pack("<f", float(b)))[0] if mode == MODE_NUMBER else 0
    dm_args = {"num_tiles": counts, "start_id": starts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values={**dm_args, **geom_args}),
            E.KernelRunArgs(
                kernel="compute",
                runtime_arg_values={**dm_args, "number_bits": per_core(number_bits), **geom_args},
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, indices


def io_tensors(a, b, out):
    """The tensor list of the `generic_op` call matching `create_program_artifacts`' indices."""
    return [a, out] if _is_number(b) else [a, b, out]

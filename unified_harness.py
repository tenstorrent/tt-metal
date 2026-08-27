# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host harness for running unified kernels through ``ttnn.generic_op``.

A unified kernel is ONE source file that describes a whole Tensix pipeline. The
trick that makes it work on today's API is that ``ttnn.generic_op`` takes a
*list* of ``KernelDescriptor``s, and nothing requires them to point at different
files. So we emit three descriptors from one source:

    reader   DataMovementConfigDescriptor(RISCV_1)   -> COMPILE_FOR_NCRISC
    writer   DataMovementConfigDescriptor(RISCV_0)   -> COMPILE_FOR_BRISC
    compute  ComputeConfigDescriptor()               -> UCK_CHLKC_{UNPACK,MATH,PACK}

No per-thread defines are passed from here. Metal already emits a thread
identity define for every kernel build, and ``unified_metal.hpp`` derives the
projection from it -- so the host side stays ignorant of the mechanism.

All three descriptors get *identical* compile-time and runtime args, because
they are the same source reading the same arg indices. That uniform arg schema
is a property of the model, not a limitation of the harness.
"""

import os

import ttnn

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.dirname(os.path.abspath(__file__)))

# The unified headers live at the repo root; the kernel's quoted includes need
# that on the JIT include path.
UNIFIED_INCLUDE_PATHS = [TT_METAL_HOME]

TILE_HW = 32 * 32
# bfloat8_b is a BLOCK format: 1024 mantissa bytes plus a 64-byte exponent section per
# tile, so it is 1088 rather than the 1024 a naive one-byte-per-element count would give.
DTYPE_TILE_BYTES = {
    ttnn.bfloat16: TILE_HW * 2,
    ttnn.float32: TILE_HW * 4,
    ttnn.bfloat8_b: TILE_HW + 64,
}


# Per data-movement thread: two for the multicast handshake, then one more for
# the arrival flag a multicast noc_core_write raises on its receivers. Laid out as
# [ready0, sent0, ready1, sent1, copy0, copy1] -- tt/unified/api.h derives every id
# from the base, so the two groups must stay in this order.
MCAST_SEMAPHORES = 6


# Appended after the last runtime argument on every core. A kernel that names the count it
# expects then catches a launcher passing the wrong number -- the failure that has hung this
# device three times. Must match u::kRuntimeArgSentinel in tt/unified/api.h.
RUNTIME_ARG_SENTINEL = 0x5EA15EA1


def make_runtime_args(cores, values):
    """A RuntimeArgs over `cores`, with the sentinel appended.

    `values` is either one flat sequence, used on every core, or a dict keyed by
    CoreCoord for per-core args (a multicast sender needs to know it is the
    sender, and each core needs its own output slice).
    """
    args = ttnn.RuntimeArgs()
    if isinstance(values, dict):
        for core in cores:
            args[core.x][core.y] = list(values[core]) + [RUNTIME_ARG_SENTINEL]
    else:
        for core in cores:
            args[core.x][core.y] = list(values) + [RUNTIME_ARG_SENTINEL]
    return args


def make_cb(cb_index, core_ranges, dtype=ttnn.bfloat16, num_pages=2):
    """A double-buffered CB by default: `num_pages` pages of one tile each."""
    page_size = DTYPE_TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=dtype, page_size=page_size)],
    )


def make_semaphore(sem_id, core_ranges, initial_value=0, core_type=None):
    """Reserve semaphore slot `sem_id` on every core in `core_ranges`.

    `sem_id` is an INPUT, not an index into the list you pass to
    unified_program(): the runtime honours SemaphoreDescriptor.id directly
    (program.cpp add_semaphore). It defaults to 0 in the descriptor, so two
    semaphores built without distinct ids silently land on the same slot. The
    kernel passes the matching id to u::Semaphore.

    Host-side allocation is the point: every core resolves an id to the same L1
    offset, independent of which RISC is running, and the runtime stamps
    initial_value on every launch. A kernel-owned counter gets neither.
    """
    return ttnn.SemaphoreDescriptor(
        id=sem_id,
        core_type=core_type if core_type is not None else ttnn.CoreType.WORKER,
        core_ranges=core_ranges,
        initial_value=initial_value,
    )


def unified_program(
    *,
    kernel_source,
    core_ranges,
    cores,
    cbs,
    compile_time_args,
    runtime_args,
    named_compile_time_args=None,
    reader_processor=ttnn.DataMovementProcessor.RISCV_1,
    writer_processor=ttnn.DataMovementProcessor.RISCV_0,
    semaphores=None,
    defines=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    dynamic_noc=False,
):
    """Build a ProgramDescriptor that compiles ONE source for all five threads.

    Args:
        kernel_source: path to the unified kernel, relative to TT_METAL_HOME.
        core_ranges: ttnn.CoreRangeSet the program runs on.
        cores: the individual CoreCoords, for per-core runtime args.
        cbs: list of ttnn.CBDescriptor.
        compile_time_args: list of ints, shared by all three descriptors. Positional, and
            what TensorAccessorArgs consumes -- a contiguous block whose length depends on
            each tensor's layout, which a name cannot express.
        named_compile_time_args: list of (name, value), also shared. Every SCALAR belongs
            here: a name that does not exist is a build failure, and keeping scalars out of
            the positional list is what stops the accessor offsets drifting. See
            unified_named_args_spec.md.
        runtime_args: list of ints, shared by all three descriptors.
        reader_processor / writer_processor: which RISC-V runs which DM role.
            Metal's convention is RISCV_1 for readers, RISCV_0 for writers.
        semaphores: optional list of ttnn.SemaphoreDescriptor (see make_semaphore).
            Four more are appended for the multicast handshake, above your ids.
        defines: optional extra (name, value) pairs, applied to all three.
        math_fidelity / math_approx_mode: compute config for the TRISCs.  The
            metal defaults (HiFi4, exact) are the most accurate and the slowest;
            HiFi2 halves the FPU passes per bfloat16 matmul and approx mode
            picks the cheap SFPU transcendentals.
        dynamic_noc: put both data-movement kernels in DM_DYNAMIC_NOC. Only needed to let
            a thread issue on a NOC other than its own -- in the default DM_DEDICATED_NOC
            that is a device hang, not a slowdown. Costs ~2.7% when unused, so leave it off.
    """
    # Reserve the multicast handshake semaphores: two per DM thread, placed ABOVE
    # any id the caller used so their choices stay unconstrained. Their base goes
    # to the kernel as a define; tt/unified/api.h derives each thread's ids from
    # it. Six slots out of NUM_SEMAPHORES = 16.
    user_semaphores = list(semaphores or [])
    mcast_sem_base = 1 + max((s.id for s in user_semaphores), default=-1)
    all_semaphores = user_semaphores + [
        make_semaphore(mcast_sem_base + i, core_ranges, initial_value=0) for i in range(MCAST_SEMAPHORES)
    ]
    # The core grid, so synchronize_cores() can default to the whole program.
    # Bounding box, not num_cores: a barrier addresses a rectangle.
    bbox = core_ranges.bounding_box()
    grid_h = bbox.end.y - bbox.start.y + 1
    grid_w = bbox.end.x - bbox.start.x + 1

    # Whether the cores FILL that bounding box. core_block(12) is eight cores in row 0 and
    # four in row 1, whose box is 2x8 -- so a barrier derived from the box would address
    # four cores that were never launched and wait on them forever. Defining this only when
    # the two agree is what makes the no-region synchronize_cores() a compile error rather
    # than a hang; see unified_api_hazards.md.
    grid_exact = core_ranges.num_cores() == grid_h * grid_w

    defines = (
        list(defines or [])
        + [
            ("TT_UNIFIED_MCAST_SEM_BASE", str(mcast_sem_base)),
            ("TT_UNIFIED_CORE_GRID_H", str(grid_h)),
            ("TT_UNIFIED_CORE_GRID_W", str(grid_w)),
        ]
        + ([("TT_UNIFIED_CORE_GRID_EXACT", "1")] if grid_exact else [])
    )

    shared = dict(
        kernel_source=kernel_source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=list(compile_time_args),
        named_compile_time_args=list(named_compile_time_args or []),
        defines=defines,
        compiler_include_paths=UNIFIED_INCLUDE_PATHS,
    )

    # DM_DEDICATED_NOC tracks issued reads in a per-RISC software counter and compares it
    # against the per-core hardware response register, so it is only correct while exactly
    # one RISC issues on a given NOC. DM_DYNAMIC_NOC keeps those counters in shared L1 and
    # sums both RISCs', which is what lets two RISCs share a NOC and still barrier
    # correctly. Metal requires every DM kernel on a core to agree on the mode, so this is
    # per-program, not per-call. See unified_explicit_noc_spec.md.
    noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if dynamic_noc else ttnn.NOC_MODE.DM_DEDICATED_NOC

    kernels = [
        # reader. `noc` must be set explicitly: DataMovementConfig defaults it to
        # RISCV_0_default (= NOC 0) regardless of processor, so leaving it out puts
        # BOTH data-movement kernels on NOC 0. That costs the second NOC's
        # bandwidth, and it makes the end-of-kernel NOC-idle asserts in brisck.cc
        # cross-talk: NOC_INDEX is emitted from this field (kernel.cpp:252), so
        # BRISC would check NOC 0's read counters against its own issued count and
        # trip on reads NCRISC issued.
        ttnn.KernelDescriptor(
            **shared,
            runtime_args=make_runtime_args(cores, runtime_args),
            config=ttnn.DataMovementConfigDescriptor(
                processor=reader_processor, noc=ttnn.NOC.RISCV_1_default, noc_mode=noc_mode
            ),
        ),
        # writer
        ttnn.KernelDescriptor(
            **shared,
            runtime_args=make_runtime_args(cores, runtime_args),
            config=ttnn.DataMovementConfigDescriptor(
                processor=writer_processor, noc=ttnn.NOC.RISCV_0_default, noc_mode=noc_mode
            ),
        ),
        # compute (fans out to TRISC 0/1/2)
        ttnn.KernelDescriptor(
            **shared,
            runtime_args=make_runtime_args(cores, runtime_args),
            config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, math_approx_mode=math_approx_mode),
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=all_semaphores, cbs=list(cbs))


def single_core():
    """The (0,0) core, as (core_ranges, cores)."""
    c = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]), [c]


def core_block(n, width=8):
    """The first `n` cores in row-major order, as (core_ranges, cores).

    Row-major over a `width`-wide grid, so n=8 is one row and n=12 is a full row plus
    four. The ranges are whole rows where possible and one partial row at the end, rather
    than n single-core ranges, because a circular buffer is allocated per range.

    The `cores` order IS the partition order for per-core runtime args: cores[i] is unit i
    of whatever the caller is splitting up.
    """
    assert n >= 1
    full, rem = divmod(n, width)
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(width - 1, y)) for y in range(full)]
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    cores = [ttnn.CoreCoord(i % width, i // width) for i in range(n)]
    return ttnn.CoreRangeSet(ranges), cores


def split_evenly(total, parts):
    """`total` units over `parts` workers as [(begin, count), ...], largest parts first.

    The remainder is spread one unit per worker rather than piled on the last one: 10 over
    4 is 3, 3, 2, 2 and not 3, 3, 3, 1. Since the workers run concurrently, the makespan is
    the LARGEST share, so spreading the remainder is what keeps that at ceil(total/parts)
    instead of letting one worker fall behind. Workers past `total` get a count of zero.
    """
    base, rem = divmod(total, parts)
    out, begin = [], 0
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        out.append((begin, count))
        begin += count
    return out


# ---------------------------------------------------------------------------
# Metal 2.0 path
#
# The same model, built as a ProgramSpec instead of a ProgramDescriptor. Everything above
# stays; a suite moves over one at a time and the two paths coexist. See
# unified_metal2_spec.md.
#
# THE ONE THING THAT IS GENUINELY NEW is that the host now has to say, per buffer, which
# kernel produces it and which consumes it. Metal 2.0 requires exactly one of each per node
# (dataflow_buffer_spec.hpp), and it refuses a program where a buffer has neither.
#
# That is not a new fact about the model, though -- it is the table at the top of
# tt/unified/api.h, which has always said the same thing in a comment:
#
#     INPUT                    OUTPUT                   INTERMED
#          DM    Compute            DM    Compute            DM    Compute
#     reserve <- *               * -> reserve                   reserve
#       write                          write                      write
#        push ->    wait         wait <-  push                     push
#                  read          read                              wait
#           * <-     pop          pop -> *                         read
#                                                                   pop
#
# INPUT means the DM thread produces and compute consumes; OUTPUT is the other way round;
# INTERMED is compute on both ends, which Metal 2.0 supports as a "self-loop" DFB
# (program_spec.cpp:942). So the port turns that comment into something the host validates.
#
# What the comment does NOT say, and the host now needs, is WHICH data-movement thread. That
# lives in the kernel today, in the `thread` template argument of every noc_load / noc_store,
# and there is nothing checking the two agree. A mismatch is not silent -- the buffer's
# endpoint lands on the wrong KernelSpec, so either validation refuses the spec or the kernel
# names an accessor that was never bound -- but it is a new contract, and the kernel-side
# static_assert described in unified_kernels/unary.cpp is what closes the other half of it.
# ---------------------------------------------------------------------------

# Kernel DM thread number -> the RISC that runs it. This is adaptor_v1.hpp's mapping, read
# in the other direction: it derives the thread id from COMPILE_FOR_BRISC / COMPILE_FOR_NCRISC,
# so the host has to place the KernelSpecs to match or `noc_load<0>` runs on the wrong core.
DM_THREAD_PROCESSOR = {
    0: (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
    1: (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
}


class Dfb:
    """A dataflow buffer, plus which projections stand at its two ends.

    `kind` is one of the three columns of the api.h table; `thread` is the DM thread at the
    data-movement end, and is None for an INTERMED buffer, which has no data-movement end.
    """

    INPUT = "input"
    OUTPUT = "output"
    INTERMED = "intermed"

    def __init__(self, name, kind, thread, dtype, num_pages):
        self.name = name
        self.kind = kind
        self.thread = thread
        self.dtype = dtype
        self.num_pages = num_pages


def dfb_input(name, thread, *, dtype=ttnn.bfloat16, num_pages=2):
    """Filled by DM thread `thread`, read by compute."""
    return Dfb(name, Dfb.INPUT, thread, dtype, num_pages)


def dfb_output(name, thread, *, dtype=ttnn.bfloat16, num_pages=2):
    """Filled by compute, drained by DM thread `thread`."""
    return Dfb(name, Dfb.OUTPUT, thread, dtype, num_pages)


def dfb_intermed(name, *, dtype=ttnn.bfloat16, num_pages=2):
    """Compute on both ends: an accumulator, a retained value, a scratch block."""
    return Dfb(name, Dfb.INTERMED, None, dtype, num_pages)


def unified_program_spec(
    *,
    kernel_source,
    nodes,
    dfbs,
    tensors,
    named_compile_time_args=None,
    defines=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    dynamic_noc=False,
    name="unified",
):
    """Build a ProgramSpec that compiles ONE source for all five threads.

    The Metal 2.0 counterpart of unified_program(). Same idea -- three KernelSpecs pointing
    at one file, identical compile-time args on all three -- with three differences that the
    2.0 host API forces rather than invites:

      * Compile-time args are NAMED ONLY. KernelSpec::CompileTimeArgs is a
        Table<string, uint32_t>; there is no positional list, so there is nowhere for a
        TensorAccessorArgs block to go and nothing for its offsets to drift against.
      * Tensors are BOUND, not addressed. Each gets a TensorParameter and a per-kernel
        accessor name, and the kernel says TensorAccessor(tensor::<name>). No base address
        runtime arg, so hazard D18 has no surface left.
      * Buffers carry ENDPOINT ROLES, which is the new obligation; see above.

    Args:
        kernel_source: path to the unified kernel, relative to TT_METAL_HOME.
        nodes: a ttnn.CoreRangeSet (or CoreCoord / CoreRange) the program runs on.
        dfbs: list of Dfb, from dfb_input / dfb_output / dfb_intermed. DECLARATION ORDER
            MATTERS -- see the slot note below.
        tensors: dict of {parameter name: ttnn.Tensor}. Every kernel binds every tensor,
            which is legal because a tensor binding carries no exclusive role (unlike a DFB
            binding), and necessary because a unified kernel names every accessor on every
            projection.
        named_compile_time_args: list of (name, value), shared by all three kernels. The
            buffer slots are appended to this, so a kernel reads its own circular buffer id
            by name rather than hardcoding a number.

    Returns the ProgramSpec. Run it with run_unified_spec().

    ON BUFFER SLOTS. A kernel needs its buffers' slot numbers as compile-time VALUES, not as
    `dfb::` binding tokens: a token is emitted only into the kernels that bind that buffer
    (genfiles.cpp:129), a DFB's two endpoint roles are both spoken for, and a unified kernel
    declares every Storage on every projection -- so a token spelling does not compile. See
    unified_gate/gate_a_tokens.cpp, which fails for exactly that reason.

    So the slot is passed as a named compile-time arg, and it is PREDICTED here from metal's
    allocator rule: the lowest free slot among buffers sharing cores, in declaration order
    (dataflow_buffer.cpp:1724). Since every buffer here shares the whole node set, that is
    just 0, 1, 2, ... in the order given.

    A prediction is not a guarantee, so it is CHECKED rather than trusted -- on the compute
    projection, which is the one projection that binds every buffer (inputs as consumer,
    outputs as producer, intermediates as both) and can therefore compare all of them against
    the tokens the host really assigned. That check is a static_assert in the kernel; see
    unified_kernels/unary.cpp.
    """
    ps = ttnn.program_spec

    # The slot prediction. Kept in one place so the assumption is stated once rather than in
    # every kernel, and so the kernel-side check has something to check against.
    slots = {d.name: i for i, d in enumerate(dfbs)}

    named_cts = list(named_compile_time_args or [])
    named_cts += [(f"cb_{d.name}", slots[d.name]) for d in dfbs]

    bbox = nodes.bounding_box() if hasattr(nodes, "bounding_box") else None
    grid_h = (bbox.end.y - bbox.start.y + 1) if bbox else 1
    grid_w = (bbox.end.x - bbox.start.x + 1) if bbox else 1
    all_defines = list(defines or []) + [
        ("TT_UNIFIED_CORE_GRID_H", str(grid_h)),
        ("TT_UNIFIED_CORE_GRID_W", str(grid_w)),
    ]
    if bbox is not None and nodes.num_cores() == grid_h * grid_w:
        all_defines.append(("TT_UNIFIED_CORE_GRID_EXACT", "1"))

    def make_kernel(unique_id, hw_config):
        k = ps.KernelSpec()
        k.unique_id = unique_id
        k.source = kernel_source
        k.hw_config = hw_config
        k.compile_time_args = {n: int(v) for n, v in named_cts}
        k.compiler_options.include_paths = UNIFIED_INCLUDE_PATHS
        k.compiler_options.defines = {n: v for n, v in all_defines}
        bindings = []
        for param in tensors:
            b = ps.KernelSpec.TensorBinding()
            b.tensor_parameter_name = param
            b.accessor_name = param
            bindings.append(b)
        k.tensor_bindings = bindings
        return k

    dm_cfgs = {}
    noc_mode = None  # DataMovementGen1Config's own default unless dynamic_noc is asked for
    for thread, (proc, noc) in DM_THREAD_PROCESSOR.items():
        cfg = ps.DataMovementGen1Config()
        cfg.processor = proc
        cfg.noc = noc
        if dynamic_noc:
            cfg.noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC
        dm_cfgs[thread] = cfg

    compute_cfg = ps.ComputeGen1Config()
    compute_cfg.fpu_math_fidelity = math_fidelity
    compute_cfg.sfpu_precision_mode = ps.Precision.Approximate if math_approx_mode else ps.Precision.Precise

    kernels = {
        "dm0": make_kernel("dm0", dm_cfgs[0]),
        "dm1": make_kernel("dm1", dm_cfgs[1]),
        "compute": make_kernel("compute", compute_cfg),
    }

    # The endpoint bindings: the api.h table, applied.
    dfb_specs = []
    for d in dfbs:
        spec = ps.DataflowBufferSpec()
        spec.unique_id = d.name
        spec.entry_size = DTYPE_TILE_BYTES[d.dtype]
        spec.num_entries = d.num_pages
        spec.data_format_metadata = d.dtype
        dfb_specs.append(spec)

        if d.kind == Dfb.INPUT:
            kernels[f"dm{d.thread}"].dfb_bindings += [ps.producer_of(d.name, d.name)]
            kernels["compute"].dfb_bindings += [ps.consumer_of(d.name, d.name)]
        elif d.kind == Dfb.OUTPUT:
            kernels["compute"].dfb_bindings += [ps.producer_of(d.name, d.name)]
            kernels[f"dm{d.thread}"].dfb_bindings += [ps.consumer_of(d.name, d.name)]
        elif d.kind == Dfb.INTERMED:
            # A self-loop: compute stands at both ends. Two bindings on ONE kernel, which
            # Metal 2.0 allows precisely because their roles are opposite.
            kernels["compute"].dfb_bindings += [
                ps.producer_of(d.name, d.name),
                ps.consumer_of(d.name, d.name),
            ]
        else:
            raise ValueError(f"unknown dfb kind {d.kind!r}")

    wu = ps.WorkUnitSpec()
    wu.name = "wu0"
    wu.kernels = ["dm0", "dm1", "compute"]
    wu.target_nodes = nodes

    spec = ps.ProgramSpec()
    spec.name = name
    spec.kernels = [kernels["dm0"], kernels["dm1"], kernels["compute"]]
    spec.dataflow_buffers = dfb_specs
    spec.tensor_parameters = [ps.TensorParameter(n, t.spec) for n, t in tensors.items()]
    spec.work_units = [wu]
    return spec


def run_unified_spec(device, spec, tensors):
    """Build the program from `spec`, bind `tensors`, enqueue it, and wait.

    `tensors` is the same dict handed to unified_program_spec, so the TensorParameters and
    the TensorArguments cannot drift apart.
    """
    ps = ttnn.program_spec
    ps.run_program_spec(device, spec, ps.ProgramRunArgs(), list(tensors.items()))

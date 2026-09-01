# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host harness for running unified kernels through a Metal 2.0 ProgramSpec.

A unified kernel is ONE source file that describes a whole Tensix pipeline. What makes that
work is that a ProgramSpec takes a set of KernelSpecs and nothing requires them to point at
different files, so we emit three from one source:

    dm0      DataMovementGen1Config(RISCV_0)  -> COMPILE_FOR_BRISC   -> DM thread 0
    dm1      DataMovementGen1Config(RISCV_1)  -> COMPILE_FOR_NCRISC  -> DM thread 1
    compute  ComputeGen1Config()              -> UCK_CHLKC_{UNPACK,MATH,PACK}

No per-thread defines are passed from here. Metal already emits a thread identity define for
every kernel build and tt/unified/adaptor.hpp derives the projection from it, so the host
stays ignorant of the mechanism.

All three get IDENTICAL compile-time args, runtime-argument schemas and tensor bindings,
because they are the same source reading the same names. That uniform schema is a property of
the model, not a limitation of the harness.

The descriptor path this used to build (ttnn.generic_op + ProgramDescriptor) is gone; see
unified_metal2_spec.md for what the move bought and what it cost.
"""

import os
import re

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


def _stable(value):
    """A rendering of `value` that depends on its CONTENT and not on where it lives.

    repr() is the fallback, and repr() of a plain object embeds its address -- which would make
    every key unique and the cache miss forever, silently. So the known argument types are
    rendered by hand and _cache_key_of refuses anything that still looks like an address.
    """
    if isinstance(value, Dfb):
        return ("dfb", value.name, value.num_entries, str(value.dtype), value.kind, value.thread)
    if isinstance(value, (list, tuple)):
        return tuple(_stable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _stable(v)) for k, v in value.items()))
    if isinstance(value, ttnn.Tensor):
        # The SPEC, not the tensor: the ProgramSpec uses only the layout, and the address
        # arrives per launch as a tensor argument. Keying on identity would never hit. And the
        # spec's own repr is an address too, so it is spelled out field by field -- these four
        # are what TensorParameter is built from.
        spec = value.spec
        return ("tensor_spec", str(spec.shape), str(spec.dtype), str(spec.layout), str(spec.memory_config))
    return repr(value)


def _cache_key_of(args):
    """The cache key for a set of unified_program_spec arguments.

    Built from EVERY argument rather than a list someone maintains: a key that misses a field
    is a silent wrong-program hit, which is the worst failure a cache can have, and the way not
    to miss one is never to enumerate them.
    """
    key = repr(sorted((name, _stable(value)) for name, value in args.items() if name != "ps"))
    # An address in the key means some argument fell through to repr() and the cache would miss
    # on every call -- which is safe, useless, and invisible. Better to say so.
    assert "object at 0x" not in key, f"unstable cache key; a value needs a _stable() case: {key}"
    return key


# spec object -> the cache key unified_program_spec derived for it. A side table rather than an
# attribute because ProgramSpec is a bound C++ type with no room for one. Keyed by id(), which
# is safe only while the spec is alive; every caller holds its spec across the run that uses it.
_CACHE_KEYS = {}


def single_core():
    """The (0,0) core, as (core_ranges, cores)."""
    c = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]), [c]


def core_block(n, width=8):
    """The first `n` cores in row-major order, as (core_ranges, cores).

    Row-major over a `width`-wide grid, so n=8 is one row and n=12 is a full row plus
    four. The ranges are whole rows where possible and one partial row at the end, rather
    than n single-core ranges, because a dataflow buffer is allocated per range.

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

# Kernel DM thread number -> the RISC that runs it. This is adaptor.hpp's mapping, read
# in the other direction: it derives the thread id from COMPILE_FOR_BRISC / COMPILE_FOR_NCRISC,
# so the host has to place the KernelSpecs to match or `noc_load<0>` runs on the wrong core.
DM_THREAD_PROCESSOR = {
    0: (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
    1: (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
}


class Dfb:
    """A dataflow buffer: its name, its depth, and which projections stand at its two ends.

    `kind` is one of the three columns of the api.h table; `thread` is the DM thread at the
    data-movement end, and is None for an INTERMED buffer, which has no data-movement end.
    Both are DERIVED FROM THE KERNEL by default -- see derive_roles() for why.
    """

    INPUT = "input"
    OUTPUT = "output"
    INTERMED = "intermed"

    def __init__(self, name, num_entries=2, *, dtype=ttnn.bfloat16, kind=None, thread=None):
        self.name = name
        self.num_entries = num_entries
        self.dtype = dtype
        self.kind = kind
        self.thread = thread


def dfb(name, num_entries=2, *, dtype=ttnn.bfloat16):
    """A buffer whose endpoints the harness reads off the kernel. The usual form."""
    return Dfb(name, num_entries, dtype=dtype)


def dfb_input(name, thread, *, dtype=ttnn.bfloat16, num_entries=2):
    """Filled by DM thread `thread`, read by compute. Stated rather than derived."""
    return Dfb(name, num_entries, dtype=dtype, kind=Dfb.INPUT, thread=thread)


def dfb_output(name, thread, *, dtype=ttnn.bfloat16, num_entries=2):
    """Filled by compute, drained by DM thread `thread`. Stated rather than derived."""
    return Dfb(name, num_entries, dtype=dtype, kind=Dfb.OUTPUT, thread=thread)


def dfb_intermed(name, *, dtype=ttnn.bfloat16, num_entries=2):
    """Compute on both ends: an accumulator, a retained value, a scratch block."""
    return Dfb(name, num_entries, dtype=dtype, kind=Dfb.INTERMED, thread=None)


# tt/unified/api.h's Storage declarations and data-movement calls, as patterns. The kernel is
# the only place that knows which projection stands at which end of a buffer, so it is the
# place to read it from.
_RE_STORAGE = re.compile(r"u::Storage<[^;]*?>\s+(\w+)\s*\(\s*(\w+)\s*\)")
_RE_DFB_NAMED = re.compile(r"(\w+)\s*=\s*get_arg\(\s*args::dfb_(\w+)\s*\)")
# A Storage built straight from the argument rather than through a named constant:
#   u::Storage<S> s1(get_arg(args::dfb_s1));
_RE_STORAGE_INLINE = re.compile(r"u::Storage<[^;]*?>\s+(\w+)\s*\(\s*get_arg\(\s*args::dfb_(\w+)\s*\)\s*\)")


# u::ComputeBlock<Shape, kDfbFoo> name = <expression>;
#
# The shim form: a compute intermediate names its buffer in its own declaration instead of
# through a u::Storage. It is scanned rather than matched by a regex because the SHAPE may
# itself carry angle brackets (u::Shape<2, 4>), and the id is the last top-level template
# argument -- a non-greedy `[^;]*?>` stops at the wrong bracket and would read the shape's
# `4` as a buffer id. Silently, which is the failure this whole function exists to avoid.
_RE_COMPUTE_BLOCK_OPEN = re.compile(r"u::ComputeBlock\s*<")


def _compute_block_dfbs(src):
    """Yield (variable, dfb constant) for every shim declaration in `src`."""
    for m in _RE_COMPUTE_BLOCK_OPEN.finditer(src):
        i = m.end()  # just past the '<'
        depth, args, start = 1, [], i
        while i < len(src) and depth:
            c = src[i]
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if not depth:
                    args.append(src[start:i])
            elif c == "," and depth == 1:
                args.append(src[start:i])
                start = i + 1
            elif c == ";":
                break  # ran past the declaration; not a shim
            i += 1
        if depth or len(args) != 2:
            continue  # u::ComputeBlock<S> or something unparsed -- not the shim form
        tail = re.match(r"\s+(\w+)\s*=", src[i:])
        if tail:
            yield tail.group(1), args[1].strip()


_RE_PRODUCED = re.compile(
    r"(?:u::Block[\w<>: ]*|u::RetainedBlock[\w<>: ]*|auto)\s+(\w+)\s*=\s*(\w+)\.(?:store|accumulate)\("
)
_RE_FILLS = re.compile(r"(?:noc_load|fill_reduce_scaler|noc_core_write|noc_core_read)<([^>(]*)>\(\s*(\w+)")
_RE_DRAINS = re.compile(r"noc_store<([^>(]*)>\(\s*std::move\((\w+)\)")
_RE_DRAINS_C2C = re.compile(r"noc_core_write<([^>(]*)>\(\s*\w+\s*,\s*std::move\((\w+)\)")
# The inline form, which is the commoner one: noc_store<1>(out_storage.store(...), out, c).
_RE_DRAINS_INLINE = re.compile(r"noc_store<([^>(]*)>\(\s*(\w+)\.(?:store|accumulate)\(")
# An Accumulator's finishing Block belongs to its OUTPUT storage, which is its second
# constructor argument -- the first is the accumulation buffer the next call re-consumes.
_RE_ACCUM = re.compile(r"u::Accumulator<[^;]*?>\s+(\w+)\s*\(\s*\w+\s*,\s*(\w+)\s*\)")
# custom_compute's escape hatch: the routine did the reserve/pack/push itself and hands the
# harness a bare handle. u::Block<Blk>{out_storage} names its Storage directly.
_RE_DRAINS_BARE = re.compile(r"noc_store<([^>(]*)>\(\s*u::Block<\w+>\s*\{\s*(\w+)\s*\}")


_RE_DEFAULT = re.compile(r"#ifndef\s+(\w+)\s*\n#define\s+\1\s+(\d+)")


def _thread_of(text, defines, kernel_source):
    """The `thread` template argument of a data-movement call, as an int.

    It is often a define -- MC_DM_THREAD, MMB_IN0_THREAD -- because which thread drives a
    transfer is a thing launchers tune. The harness is passing those defines, so it can
    resolve them; anything it cannot resolve is an error rather than a guess.
    """
    tok = text.split(",")[0].strip()
    if tok.isdigit():
        return int(tok)
    if tok in defines:
        return int(defines[tok])
    # A launcher that does not override the knob gets the kernel's own default, which the
    # kernel states as `#ifndef X / #define X n`.
    src = open(os.path.join(TT_METAL_HOME, kernel_source)).read()
    for name, value in _RE_DEFAULT.findall(src):
        if name == tok:
            return int(value)
    raise ValueError(
        f"{kernel_source} names its data-movement thread as `{tok}`, which is not a literal and "
        f"not among the defines this launcher passes, so the buffer's endpoint cannot be resolved."
    )


def derive_roles(kernel_source, defines):
    """Work out each buffer's endpoints by reading the kernel.

    WHY THIS IS NOT A LAUNCHER'S JOB. Metal 2.0 wants a producer and a consumer named per
    buffer, and the DM half of that is a thread number the KERNEL already states, in the
    `thread` argument of every noc_load / noc_store. Having a launcher restate it creates a
    contract with two ends and nothing between them -- and the mismatch is SILENT on Gen1.
    Verified: binding `out` to thread 0 while the kernel stores it on thread 1 runs, and
    passes, bit-identical. Gen1 dataflow-buffer state is per core rather than per RISC, so
    either data-movement kernel can drive it whatever the host declared; the endpoint masks
    only become load-bearing on Gen2, where they drive the tile counters.

    So a wrong thread here is invisible today and a hang on the next architecture, which is
    the worst shape a hazard can have. Deriving it removes the second end of the contract
    rather than trying to check it.

    Returns {buffer name: (kind, thread)}. Raises on anything it cannot read.
    """
    src = open(os.path.join(TT_METAL_HOME, kernel_source)).read()
    # kDfbFoo -> "foo" (from its dfb_<name> compile-time arg), then Storage variable -> "foo".
    dfb_const = dict(_RE_DFB_NAMED.findall(src))
    storages = {}
    for var, dfb in _RE_STORAGE.findall(src):
        if dfb in dfb_const:
            storages[var] = dfb_const[dfb]
    for var, name in _RE_STORAGE_INLINE.findall(src):
        storages[var] = name
    # A shim-declared buffer is a compute intermediate by construction: it has no Storage,
    # so no noc call can name it, so the role lookup below can only ever land on INTERMED.
    for var, dfb in _compute_block_dfbs(src):
        if dfb in dfb_const:
            storages[var] = dfb_const[dfb]
    # Block variable -> every Storage it could have come from. A name is reused across
    # preprocessor branches (`u::Block result` in each), so this is one-to-many rather than
    # one-to-one, and every candidate gets the role: they all drain the same way.
    accum_out = dict(_RE_ACCUM.findall(src))
    produced = {}
    for block, holder in _RE_PRODUCED.findall(src):
        produced.setdefault(block, set()).add(accum_out.get(holder, holder))

    roles = {}
    for thread, var in _RE_FILLS.findall(src):
        roles.setdefault(var, (Dfb.INPUT, _thread_of(thread, defines, kernel_source)))
    for thread, storage in _RE_DRAINS_BARE.findall(src):
        roles.setdefault(storage, (Dfb.OUTPUT, _thread_of(thread, defines, kernel_source)))
    for thread, holder in _RE_DRAINS_INLINE.findall(src):
        roles.setdefault(accum_out.get(holder, holder), (Dfb.OUTPUT, _thread_of(thread, defines, kernel_source)))
    for pattern in (_RE_DRAINS, _RE_DRAINS_C2C):
        for thread, block in pattern.findall(src):
            for src_storage in produced.get(block, ()):
                roles.setdefault(src_storage, (Dfb.OUTPUT, _thread_of(thread, defines, kernel_source)))

    return {name: roles.get(var, (Dfb.INTERMED, None)) for var, name in storages.items()}


def unified_program_spec(
    *,
    kernel_source,
    nodes,
    dfbs,
    tensors,
    named_compile_time_args=None,
    runtime_arg_names=None,
    semaphores=None,
    defines=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    dst_32bit=False,
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
            buffer slots are appended to this, so a kernel reads its own dataflow buffer id
            by name rather than hardcoding a number.
        runtime_arg_names: names the kernel reads with get_arg(args::<name>). Declared on all
            three kernels, because the uniform argument schema is a property of the model.
            Values are supplied to run_unified_spec(), per node or broadcast. THIS IS D17:
            there is no positional list left to miscount, so the sentinel that guarded it on
            the legacy path has nothing to guard.
        semaphores: names of USER semaphores, in the order they should be allocated. Six more
            are appended for the multicast handshake and the core-to-core arrival flag, above
            whatever the caller asked for -- exactly as unified_program() does.

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

    # THE CACHE KEY, derived from every argument this function was handed rather than from a
    # list someone maintains. A key that misses a field is a silent wrong-program hit -- the
    # worst failure the cache can have -- and the way to not miss one is to never enumerate
    # them. locals() here is exactly the parameter list, taken before anything below mutates
    # it.
    #
    # Tensors reduce to their SPEC, not their identity: the ProgramSpec uses only the layout
    # (TensorParameter), and the addresses arrive per launch as tensor arguments. Keying on
    # the tensors themselves would miss on every call and cache nothing.
    cache_key = _cache_key_of(locals())

    # The slot prediction. Kept in one place so the assumption is stated once rather than in
    # every kernel, and so the kernel-side check has something to check against.
    slots = {d.name: i for i, d in enumerate(dfbs)}

    named_cts = list(named_compile_time_args or [])
    named_cts += [(f"dfb_{d.name}", slots[d.name]) for d in dfbs]

    # The reserved handshake semaphores, laid out exactly as unified_program() lays them out:
    # two per data-movement thread, then one arrival flag per thread. tt/unified/api.h derives
    # every id from the base by arithmetic, so the six must be CONTIGUOUS and in this order.
    #
    # They are allocated after the caller's, so user ids stay unconstrained; and metal assigns
    # the lowest free id among cores that share a core set (program.cpp:2021), which for one
    # core set is just declaration order. So the base is len(user semaphores) -- predicted,
    # like the buffer slots, and checked the same way: the harness passes the FIRST and LAST
    # reserved names as sem:: tokens and tt/unified/api.h static_asserts the arithmetic
    # against them. First and last together pin the whole run, since metal cannot issue a
    # duplicate id.
    user_sems = list(semaphores or [])
    reserved_sems = [
        "u_mcast_ready0",
        "u_mcast_sent0",
        "u_mcast_ready1",
        "u_mcast_sent1",
        "u_copy_arrived0",
        "u_copy_arrived1",
    ]
    assert len(reserved_sems) == MCAST_SEMAPHORES
    all_sems = user_sems + reserved_sems
    mcast_sem_base = len(user_sems)

    bbox = nodes.bounding_box() if hasattr(nodes, "bounding_box") else None
    grid_h = (bbox.end.y - bbox.start.y + 1) if bbox else 1
    grid_w = (bbox.end.x - bbox.start.x + 1) if bbox else 1
    all_defines = list(defines or []) + [
        ("TT_UNIFIED_MCAST_SEM_BASE", str(mcast_sem_base)),
        # The two ends of the reserved run, as sem:: tokens for api.h to check the base and
        # the contiguity against. Spelled as an expression rather than a value on purpose:
        # the value is the host's to know and the token is the only thing that reports it.
        ("TT_UNIFIED_MCAST_SEM_FIRST", f"sem::{reserved_sems[0]}"),
        ("TT_UNIFIED_MCAST_SEM_LAST", f"sem::{reserved_sems[-1]}"),
        ("TT_UNIFIED_CORE_GRID_H", str(grid_h)),
        ("TT_UNIFIED_CORE_GRID_W", str(grid_w)),
    ]
    if bbox is not None and nodes.num_cores() == grid_h * grid_w:
        all_defines.append(("TT_UNIFIED_CORE_GRID_EXACT", "1"))
    # The DST budget halves with a 32-bit Dest, and math.hpp's kMaxDstTiles has to know.
    # It goes to ALL THREE kernels, not just compute, because the budget picks between two
    # code paths that emit different buffer protocol -- so a value only compute could see
    # would be a hang rather than a wrong number. Metal's own DST_ACCUM_MODE is exactly
    # that: generated behind a UCK_CHLKC_* guard, invisible to data movement. Compute
    # static_asserts the two against each other, which is what keeps this honest.
    all_defines.append(("TT_UNIFIED_DST_32BIT", "1" if dst_32bit else "0"))

    def make_kernel(unique_id, hw_config, is_compute):
        k = ps.KernelSpec()
        k.unique_id = unique_id
        k.source = kernel_source
        k.hw_config = hw_config
        k.compile_time_args = {n: int(v) for n, v in named_cts}
        k.compiler_options.include_paths = UNIFIED_INCLUDE_PATHS
        # The semaphore-check tokens go only to the kernels that BIND the semaphores, for the
        # same reason the buffer slots are values and not tokens: sem::<name> exists only
        # where it was bound. Compute cannot bind one at all (see below), so it does not get
        # the check -- which costs nothing, since the two data-movement projections between
        # them check every reserved id, and all five see the same base.
        defs = dict(all_defines)
        if is_compute:
            defs.pop("TT_UNIFIED_MCAST_SEM_FIRST", None)
            defs.pop("TT_UNIFIED_MCAST_SEM_LAST", None)
        k.compiler_options.defines = defs
        # KernelSpec defaults EVERY kernel to O2. The legacy path defaulted compute to O3
        # (kernel_types.hpp:132 against :82), and the difference is not only speed: an
        # LLK-heavy compute kernel built at O2 can fail to LINK, because constant propagation
        # stops reaching the addrmod immediates and LTO reports "impossible constraint in
        # 'asm'". flash_attention does exactly that. So compute keeps the level it always had.
        k.compiler_options.opt_level = ps.KernelBuildOptLevel.O3 if is_compute else ps.KernelBuildOptLevel.O2
        bindings = []
        for param in tensors:
            b = ps.KernelSpec.TensorBinding()
            b.tensor_parameter_name = param
            b.accessor_name = param
            bindings.append(b)
        k.tensor_bindings = bindings
        # Both DATA-MOVEMENT kernels bind every semaphore, and the compute kernel binds none.
        #
        # Not a choice: Metal 2.0 refuses semaphore bindings on a compute kernel outright
        # (program_spec.cpp:1088). That turns out to agree with the model rather than fight
        # it -- tt/unified/api.h already says a Semaphore is projected onto one DM thread and
        # is a no-op elsewhere, and impl.hpp keeps metal's Semaphore behind an
        # IS_DM_THREAD guard, so compute has never touched one. The rule the model documented
        # is now the rule the host enforces.
        #
        # Both DM kernels bind all of them, rather than one each, because a semaphore binding
        # carries no exclusive role (unlike a DFB binding, whose two endpoints are spoken
        # for) and because either thread may be the one driving a given collective.
        if not is_compute:
            sem_bindings = []
            for sem_name in all_sems:
                b = ps.KernelSpec.SemaphoreBinding()
                b.semaphore_spec_name = sem_name
                b.accessor_name = sem_name
                sem_bindings.append(b)
            k.semaphore_bindings = sem_bindings
        k.runtime_arg_schema.runtime_arg_names = list(runtime_arg_names or [])
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
    compute_cfg.enable_32_bit_dest = dst_32bit

    kernels = {
        "dm0": make_kernel("dm0", dm_cfgs[0], is_compute=False),
        "dm1": make_kernel("dm1", dm_cfgs[1], is_compute=False),
        "compute": make_kernel("compute", compute_cfg, is_compute=True),
    }

    # The endpoint bindings: the api.h table, applied.
    dfb_specs = []
    derived = derive_roles(kernel_source, dict(all_defines))
    for d in dfbs:
        if d.kind is None:
            if d.name not in derived:
                raise ValueError(
                    f"{kernel_source} declares no u::Storage on a dfb_{d.name} compile-time arg, so its "
                    f"endpoints "
                    f"cannot be read off the kernel. Name it as the kernel does, or state the role "
                    f"explicitly with dfb_input/dfb_output/dfb_intermed."
                )
            d.kind, d.thread = derived[d.name]
        spec = ps.DataflowBufferSpec()
        spec.unique_id = d.name
        spec.entry_size = DTYPE_TILE_BYTES[d.dtype]
        spec.num_entries = d.num_entries
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
    sem_specs = []
    for sem_name in all_sems:
        ss = ps.SemaphoreSpec()
        ss.unique_id = sem_name
        ss.target_nodes = nodes
        sem_specs.append(ss)
    spec.semaphores = sem_specs
    spec.tensor_parameters = [ps.TensorParameter(n, t.spec) for n, t in tensors.items()]
    spec.work_units = [wu]
    _CACHE_KEYS[id(spec)] = cache_key
    return spec


def run_unified_spec(device, spec, tensors, runtime_args=None, nodes=None):
    """Build the program from `spec`, bind `tensors` and `runtime_args`, enqueue it, and wait.

    `tensors` is the same dict handed to unified_program_spec, so the TensorParameters and
    the TensorArguments cannot drift apart.

    `runtime_args` is {name: value} for a value every node shares, or {name: {CoreCoord:
    value}} where nodes differ -- a multicast sender's own coordinate, or each core's output
    slice. `nodes` is the list of CoreCoords to broadcast a shared value over, and is only
    needed when some argument is given in the scalar form.

    Names are the schema declared on the kernels; a name that is not in it, or one that is in
    it and not supplied here, is an error from metal rather than a garbage read. That is D17
    closed: there is no positional list left to get out of step.
    """
    ps = ttnn.program_spec

    per_kernel = []
    for kernel_name in ("dm0", "dm1", "compute"):
        kra = ps.ProgramRunArgs.KernelRunArgs()
        kra.kernel = kernel_name
        values = {}
        for name, value in (runtime_args or {}).items():
            if isinstance(value, dict):
                values[name] = {c: int(v) for c, v in value.items()}
            else:
                if nodes is None:
                    raise ValueError(f"runtime arg {name!r} is a scalar, so run_unified_spec needs `nodes`")
                values[name] = {c: int(value) for c in nodes}
        kra.runtime_arg_values = values
        per_kernel.append(kra)

    run_args = ps.ProgramRunArgs()
    run_args.kernel_run_args = per_kernel
    ps.run_program_spec(device, spec, run_args, list(tensors.items()), cache_key=_CACHE_KEYS.get(id(spec), ""))
